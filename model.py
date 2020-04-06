from custom.layers import *
from custom.layers import Encoder

import sys
import torch
from torch import nn
import torch.distributions as dist
import random
import utils

import torch
from tensorboardX import SummaryWriter
from progress.bar import Bar
import torch.nn.functional as F


class MusicTransformerVAE(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False, writer=None):
        super().__init__()
        self.infer = False
        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist

        self.writer = writer
        self.encoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x, length=None, writer=None):
        # if self.training or not self.infer:
        src_mask, tgt_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                                src=x, 
                                                                                trg=x)
        print(src_mask.shape, tgt_mask.shape, look_ahead_mask.shape)
        
        # encoder output
        enc_out, w = self.encoder(x, mask=look_ahead_mask)
        # enc_out = torch.mean(enc_out, dim=1).unsqueeze(1)      # mean aggregate global properties
        print(enc_out.shape)

        # decoder output
        x_padded = F.pad(input=x, pad=(1, 0, 0, 0), mode='constant', value=0)

        dec_out, w = self.decoder(x_padded[:, :-1], tgt_mask, look_ahead_mask, self.training, enc_output=enc_out)
        dec_out = self.fc(dec_out)

        # fc = self.fc(decoder)
        # return fc.contiguous() if self.training else fc.contiguous(), [weight.contiguous() for weight in w]
        return dec_out
        
        # else:
        #     return self.generate(x, length, None).contiguous().tolist()

    def generate(self,
                 prior: torch.Tensor,
                 length=2048,
                 tf_board_writer: SummaryWriter = None):
        decode_array = prior
        result_array = prior
        print(config)
        print(length)
        for i in Bar('generating').iter(range(length)):
            if decode_array.size(1) >= config.threshold_len:
                decode_array = decode_array[:, 1:]
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.size(1), decode_array, decode_array, pad_token=config.pad_token)

            # result, _ = self.forward(decode_array, lookup_mask=look_ahead_mask)
            # result, _ = decode_fn(decode_array, look_ahead_mask)
            result, _ = self.Decoder(decode_array, None)
            result = self.fc(result)
            result = result.softmax(-1)

            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            u = 0
            if u > 1:
                result = result[:, -1].argmax(-1).to(decode_array.dtype)
                decode_array = torch.cat((decode_array, result.unsqueeze(-1)), -1)
            else:
                pdf = dist.OneHotCategorical(probs=result[:, -1])
                result = pdf.sample().argmax(-1).unsqueeze(-1)
                # result = torch.transpose(result, 1, 0).to(torch.int32)
                decode_array = torch.cat((decode_array, result), dim=-1)
                result_array = torch.cat((result_array, result), dim=-1)
            del look_ahead_mask
        result_array = result_array[0]
        return result_array

    def test(self):
        self.eval()
        self.infer = True


class TransformerVAE(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False, writer=None):
        super().__init__()
        self.infer = False
        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist

        self.writer = writer
        self.encoder = CommonEncoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.decoder = CommonDecoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x, length=None, writer=None):
        # if self.training or not self.infer:
        src_mask, tgt_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x)
        
        # encoder output
        enc_out, w = self.encoder(x, mask=look_ahead_mask)

        # decoder output
        x_padded = F.pad(input=x, pad=(1, 0, 0, 0), mode='constant', value=0)

        dec_out, w = self.decoder(x_padded[:, :-1], tgt_mask, look_ahead_mask, self.training, enc_output=enc_out)
        dec_out = self.fc(dec_out)

        # fc = self.fc(decoder)
        # return fc.contiguous() if self.training else fc.contiguous(), [weight.contiguous() for weight in w]
        return dec_out
        
        # else:
        #     return self.generate(x, length, None).contiguous().tolist()