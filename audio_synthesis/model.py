import torch
from torch import nn
import numpy as np
import math as m
from torch.distributions import Normal
from torch.nn import functional as F
from layers import *
from model_utils import *
from tqdm import tqdm


class PianoTacotron(torch.nn.Module):
    def __init__(self, melspec_dim, pr_dim, prenet_sizes=[256, 128], 
                 conv_dims=[32, 32, 64, 64, 128, 128],
                 lstm_dims=128, linear_dims=128, kernel_size=3, stride=2, 
                 t_num_layers=6, t_dims=128, t_dropout=0.1, t_maxlen=1000,
                 z_dims=128, k_dims=4, r=2):

        super().__init__()

        # posterior_net is shared by q_z_u and q_z_s
        self.posterior_enc = LatentEncoder(input_dim=melspec_dim + pr_dim + k_dims,
                                            kernel_size=kernel_size,
                                            stride=stride)
        self.posterior_u_init, self.posterior_s_init = nn.Linear(linear_dims, linear_dims), \
                                                        nn.Linear(linear_dims, linear_dims)
        self.posterior_u_mu, self.posterior_u_var = nn.Linear(linear_dims, z_dims), nn.Linear(linear_dims, z_dims)
        self.posterior_s_mu, self.posterior_s_var = nn.Linear(linear_dims, z_dims), nn.Linear(linear_dims, z_dims)

        # piano roll encoder
        self.piano_roll_encoder = PianoRollEncoder(input_dim=pr_dim)
        self.proj_encoder_out = nn.Linear(512, linear_dims)

        # decoder prenet
        self.prenet = PreNet(melspec_dim, sizes=prenet_sizes)

        # transformer decoder
        self.transformer_decoder = TransformerDecoder(input_dim=prenet_sizes[-1], 
                                                      num_layers=t_num_layers, 
                                                      d_model=t_dims, 
                                                      dropout=t_dropout, 
                                                      max_len=t_maxlen)
        
        # final out linear layer
        self.linear_final = nn.Linear(t_dims, melspec_dim * r)
        self.r  = r
        self.n_component = k_dims
        self.linear_dim = linear_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=0)
    
    def encode(self, x, y, is_sup=False, z_s=None):
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        # encode z_u and z_s
        # first, infer z_s 
        q_z_dist = self.posterior_enc(torch.cat([x, y], dim=-1))
        q_z_s_dist = self.posterior_u_init(q_z_dist)
        mu_z_s, var_z_s = self.posterior_s_mu(q_z_s_dist), self.posterior_s_var(q_z_s_dist).exp_()
        z_s_predict = repar(mu_z_s, var_z_s)
        cls_z_s_logits, cls_z_s_prob = self.approx_qy_x(z_s_predict, self.mu_zs_lookup, 
                                                        self.logvar_zs_lookup, 
                                                        n_component=self.n_component)
        
        # use true labels if given
        if is_sup:
            mu_zs, var_z_s = self.mu_zs_lookup(z_s), self.logvar_zs_lookup(z_s).exp_()
            z_s_labelled = repar(mu_zs, var_z_s)
            z_s_broadcast = torch.stack([z_s_labelled] * x.shape[1], dim=1)
        else:
            z_s_broadcast = torch.stack([z_s_predict] * x.shape[1], dim=1)
        
        # encode z_u
        q_z_u_dist = self.posterior_u_init(q_z_dist)
        mu_z_u, var_z_u = self.posterior_u_mu(q_z_u_dist), self.posterior_u_var(q_z_u_dist).exp_()
        z_u = repar(mu_z_u, var_z_u)
        
        # encode piano roll
        piano_roll_features = self.piano_roll_encoder(y)

        return z_s_predict, cls_z_s_logits, cls_z_s_prob, z_u, piano_roll_features, \
                Normal(mu_z_u, var_z_u), Normal(mu_z_s, var_z_s)
    
    def decode(self, x, z_s, z_u, piano_roll_features):
        # concat piano_roll_features with z_s, z_u
        z_s_broadcast = torch.stack([z_s] * x.shape[1], dim=1)
        z_u_broadcast = torch.stack([z_u] * x.shape[1], dim=1)
        encoder_output = torch.cat([piano_roll_features, z_s_broadcast, z_u_broadcast], dim=-1)
        encoder_output = self.proj_encoder_out(encoder_output)

        # decoding part
        x_padded = F.pad(input=x, pad=(0, 0, 1, 0), mode='constant', value=0)   # <START> frame
        x_padded = x_padded[:, :-1, :]      # leave out last frame for prediction

        # decoding input should be "diluted" according to reduction factor
        dec_x_in = torch.stack([x_padded[:, i, :] for i in range(0, x_padded.shape[1], self.r)], dim=1)

        # decode prenet
        dec_x_in = self.prenet(dec_x_in)
        
        # create look_ahead_mask
        dec_look_ahead_mask = generate_square_subsequent_mask(dec_x_in.shape[1]).cuda()
        
        dec_out = self.transformer_decoder(dec_x_in, 
                                            mask=None, 
                                            lookup_mask=dec_look_ahead_mask, 
                                            is_training=True,
                                            enc_output=encoder_output)
        
        # decode multiple frames according to reduction factor
        dec_out = self.linear_final(dec_out)    # (b, t // r, n_filters * r)
        dec_out = dec_out.view(dec_out.shape[0], dec_out.shape[1] * self.r, \
                                dec_out.shape[2] // self.r)
        dec_out = dec_out[:, :-1, :]            # remove last frame to remain at 625
        return dec_out
    
    def generate(self, x, y, z_s_target):
        print(x.shape, y.shape)
        z_s_predict, z_s_prob, z_u, piano_roll_features, z_u_dist = self.encode(x, y, is_sup=False, z_s=z_s_target)

        res = torch.zeros_like(x).cuda()
        out = self.decode(res, z_s_target, z_u, piano_roll_features)
        print(res.shape, out.shape)

    
    def forward(self, x, y, is_sup=False, z_s=None):
        # x: mel-spectrogram: (b, t, n_filters)
        # y: onset piano roll: (b, t, 88)

        z_s_predict, cls_z_s_logits, cls_z_s_prob, z_u, piano_roll_features, \
            z_u_dist, z_s_dist = self.encode(x, y, is_sup=is_sup, z_s=z_s)

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        if is_sup:
            mu_z_s_labelled, var_z_s_labelled = self.mu_zs_lookup(z_s), self.logvar_zs_lookup(z_s)
            z_s_labelled = repar(mu_z_s_labelled, var_z_s_labelled)
            dec_out = self.decode(x, z_s_labelled, z_u, piano_roll_features)
        else:
            dec_out = self.decode(x, z_s_predict, z_u, piano_roll_features)
        
        return dec_out, cls_z_s_prob, z_u_dist, z_s_dist
    
    def _build_mu_lookup(self):
        mu_zs_lookup = nn.Embedding(self.n_component, self.linear_dim)
        nn.init.xavier_uniform_(mu_zs_lookup.weight)
        mu_zs_lookup.weight.requires_grad = True
        self.mu_zs_lookup = mu_zs_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_zs_lookup = nn.Embedding(self.n_component, self.linear_dim)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_zs_lookup.weight, init_logvar)
        logvar_zs_lookup.weight.requires_grad = logvar_trainable
        self.logvar_zs_lookup = logvar_zs_lookup

    def _infer_class(self, q_z):
        logLogit_qy_x, qy_x = self._approx_qy_x(q_z, self.mu_zs_lookup, 
                                                self.logvar_zs_lookup, 
                                                n_component=self.n_component)
        val, y = torch.max(qy_x, dim=1)
        return logLogit_qy_x, qy_x, y

    def approx_qy_x(self, z, mu_lookup, logvar_lookup, n_component):
        def log_gauss_lh(z, mu, logvar):
            """
            Calculate p(z|y), the likelihood of z w.r.t. a Gaussian component
            """
            llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
            llh = torch.sum(llh, dim=1)  # sum over dimensions
            return llh

        logLogit_qy_x = torch.zeros(z.shape[0], n_component).cuda()  # log-logit of q(y|x)
        for k_i in torch.arange(0, n_component):
            mu_k, logvar_k = mu_lookup(k_i.cuda()), logvar_lookup(k_i.cuda())
            logLogit_qy_x[:, k_i] = log_gauss_lh(z, mu_k, logvar_k) + np.log(1 / n_component)

        qy_x = torch.nn.functional.softmax(logLogit_qy_x, dim=1)
        return logLogit_qy_x, qy_x

        

class MelSpecReconModel(torch.nn.Module):
    def __init__(self, melspec_dim, pr_dim, prenet_sizes=[256, 128], 
                 conv_dims=[32, 32, 64, 64, 128, 128],
                 lstm_dims=128, linear_dims=128, kernel_size=3, stride=2, 
                 t_num_layers=6, t_dims=128, t_dropout=0.1, t_maxlen=1000,
                 z_dims=128, k_dims=4, r=2):

        super().__init__()
        # decoder prenet
        self.prenet = PreNet(melspec_dim, sizes=prenet_sizes)

        # use CBHG
        self.piano_roll_encoder = PianoRollEncoder(input_dim=pr_dim)
        self.proj_encoder_out = nn.Linear(256, linear_dims)

        # use proj encoder out
        # self.proj_encoder_out = nn.Linear(88, linear_dims)

        # transformer decoder
        self.transformer_decoder = TransformerDecoder(input_dim=prenet_sizes[-1], 
                                                      num_layers=t_num_layers, 
                                                      d_model=t_dims, 
                                                      dropout=t_dropout, 
                                                      max_len=t_maxlen)
        
        # final out linear layer
        self.linear_final = nn.Linear(t_dims, melspec_dim)
        self.n_component = k_dims
        self.linear_dim = linear_dims
    
    def forward(self, x, y):
        # x: mel-spectrogram: (b, t, n_filters)
        # y: onset piano roll: (b, t, 88)

        x_padded = F.pad(input=x, pad=(0, 0, 1, 0), mode='constant', value=0)   # <START> frame
        x_padded = x_padded[:, :-1, :]      # leave out last frame for prediction

        # y = self.proj_encoder_out(y)
        # y = self.piano_roll_encoder(y)
        # y = self.proj_encoder_out(y)
        y = torch.cat([y, torch.zeros((y.shape[0], y.shape[1], 40)).cuda()], dim=-1)

        # decode prenet
        dec_x_in = self.prenet(x_padded)
        
        # create look_ahead_mask
        dec_look_ahead_mask = generate_square_subsequent_mask(dec_x_in.shape[1]).cuda()
        
        dec_out = self.transformer_decoder(dec_x_in, 
                                            mask=None, 
                                            lookup_mask=dec_look_ahead_mask, 
                                            is_training=True,
                                            enc_output=y)
        
        # decode multiple frames according to reduction factor
        dec_out = self.linear_final(dec_out)
        return dec_out
    
    def generate(self, y):
        x = torch.zeros(y.shape[0], 1, 128).cuda()  # (b, 1, 128) 
        # y = self.piano_roll_encoder(y)
        # y = self.proj_encoder_out(y)
        y = torch.cat([y, torch.zeros((y.shape[0], y.shape[1], 40)).cuda()], dim=-1)

        for i in tqdm(range(y.shape[1])):
            dec_x_in = self.prenet(x)
            dec_look_ahead_mask = generate_square_subsequent_mask(i+1).cuda()

            dec_out = self.transformer_decoder(dec_x_in, 
                                                mask=None, 
                                                lookup_mask=dec_look_ahead_mask, 
                                                is_training=True,
                                                enc_output=y)
            dec_out = self.linear_final(dec_out)    # (b, i+1, 128)
            print("dec_out", dec_out.shape)
            predicted_frame = dec_out[:, -1, :].unsqueeze(1).detach()
            print(predicted_frame)
            x = torch.cat([x, predicted_frame], dim=1)
            print("x", x.shape)
        
        return x




