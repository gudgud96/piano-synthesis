import torch
from torch import nn
import numpy as np
import math as m
from torch.distributions import Normal
from torch.nn import functional as F
from layers import *
from model_utils import *


class PianoTacotron(torch.nn.Module):
    def __init__(self, melspec_dim, pr_dim, prenet_sizes=[256, 128], 
                 conv_dims=[32, 32, 64, 64, 128, 128],
                 lstm_dims=128, linear_dims=128, kernel_size=3, stride=2, 
                 t_num_layers=6, t_dims=128, t_dropout=0.1, t_maxlen=1000,
                 z_dims=128, k_dims=4, r=2):

        super().__init__()

        # posterior_net is shared by q_z_u and q_z_s
        self.posterior_u_enc = LatentEncoder(input_dim=melspec_dim + pr_dim + k_dims)
        self.posterior_s_enc = LatentEncoder(input_dim=melspec_dim + pr_dim)
        self.posterior_u_mu, self.posterior_u_var = nn.Linear(linear_dims, z_dims), nn.Linear(linear_dims, z_dims)
        self.posterior_s = nn.Linear(linear_dims, k_dims)
        self.gs = GumbelSoftmax(k_dims, k_dims)

        # piano roll encoder
        self.piano_roll_encoder = PianoRollEncoder(input_dim=pr_dim)
        self.proj_encoder_out = nn.Linear(388, linear_dims)

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
    
    def encode(self, x, y, is_sup=False, z_s=None):
        # encode z_u and z_s
        # first, infer z_s 
        q_z_s_dist = self.posterior_s_enc(torch.cat([x, y], dim=-1))
        q_z_s_dist = self.posterior_s(q_z_s_dist)
        z_s_logits, z_s_prob, z_s_predict = self.gs(q_z_s_dist, hard=True)
        
        # use true labels if given
        if is_sup:
            z_s_broadcast = torch.stack([z_s] * x.shape[1], dim=1)
        else:
            z_s_broadcast = torch.stack([z_s_predict] * x.shape[1], dim=1)
        
        q_z_u_dist = self.posterior_u_enc(torch.cat([x, y, z_s_broadcast], dim=-1))
        mu_z_u, var_z_u = self.posterior_u_mu(q_z_u_dist), self.posterior_u_var(q_z_u_dist).exp_()

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        z_u = repar(mu_z_u, var_z_u)

        # encode piano roll
        piano_roll_features = self.piano_roll_encoder(y)

        return z_s_predict, z_s_prob, z_u, piano_roll_features, Normal(mu_z_u, var_z_u)
    
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
    
    def forward(self, x, y, is_sup=False, z_s=None):
        # x: mel-spectrogram: (b, t, n_filters)
        # y: onset piano roll: (b, t, 88)

        z_s_predict, z_s_prob, z_u, piano_roll_features, z_u_dist = self.encode(x, y, is_sup=is_sup, z_s=z_s)
        
        if is_sup:
            dec_out = self.decode(x, z_s, z_u, piano_roll_features)
        else:
            dec_out = self.decode(x, z_s_predict, z_u, piano_roll_features)
        
        return dec_out, z_s_prob, z_u_dist

        




