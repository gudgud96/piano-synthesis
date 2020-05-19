import torch
from torch import nn
import numpy as np
import math as m
from torch.distributions import Normal
from torch.nn import functional as F


# GS reference:
# https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Layers.py

class GumbelSoftmax(nn.Module):
    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
     
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        #categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 
  
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


class PreNet(torch.nn.Module):
    def __init__(self, input_dim, sizes=[256, 128]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        return self.net(x)

# CBHG implementation reference: 
# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py

class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


class PianoRollEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.prenet = PreNet(input_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])
    
    def forward(self, x, input_lengths=None):
        inputs = self.prenet(x)
        return self.cbhg(inputs, input_lengths)


class LatentEncoder(torch.nn.Module):
    def __init__(self, input_dim, conv_dims=[32, 32, 64, 64, 128, 128],
                 lstm_dims=128, linear_dims=128, kernel_size=3, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, conv_dims[0], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dims[0]),
            nn.Conv2d(conv_dims[0], conv_dims[1], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dims[1]),
            nn.Conv2d(conv_dims[1], conv_dims[2], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dims[2]),
            nn.Conv2d(conv_dims[2], conv_dims[3], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dims[3]),
            nn.Conv2d(conv_dims[3], conv_dims[4], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dims[4]),
            nn.Conv2d(conv_dims[4], conv_dims[5], kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dims[5]),
        )

        # TODO: this dimension is hardcoded
        self.lstm = nn.LSTM(16, lstm_dims)
        self.linear_out = nn.Linear(lstm_dims, linear_dims)
    
    def forward(self, x):
        # unsqueeze first dimension to 1 channel
        x = x.unsqueeze(1)
        x = self.net(x)         # (b, 128, 8, 2)

        # TODO: why lstm here? but follow implementation first
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.lstm(x)[0]
        x = x[:, -1, :]
        x = nn.Tanh()(self.linear_out(x))
        return x


class PositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        x = x + torch.from_numpy(self.positional_embedding[:, :x.size(1), :]).to(x.device, dtype=x.dtype)
        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(self, input_dim, num_layers=6, d_model=512, dropout=0.1, max_len=1000):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = PositionEmbedding(input_dim, max_seq=max_len)
        self.dropout = torch.nn.Dropout(dropout)

        # Here we can change different Transformer variant, or RNN + Attention
        # Now we use vanilla Transformer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=d_model // 64)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask, lookup_mask, is_training=True, enc_output=None):
        weights = []
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # need length first in this implementation
        x = torch.transpose(x, 0, 1)
        x = self.decoder(x, memory=enc_output, memory_mask=mask, tgt_mask=lookup_mask)
        x = torch.transpose(x, 0, 1)
        return x     # (batch_size, input_seq_len, d_model)

