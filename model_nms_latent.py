import torch
from torch import nn
import numpy as np
import math as m
from torch.distributions import Normal
from torch.nn import functional as F
from tqdm import tqdm
import time


class FiLM(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.gamma = nn.Linear(input_dims, output_dims)
        self.beta = nn.Linear(input_dims, output_dims)
    
    def forward(self, x, t):
        gamma, beta = self.gamma(t), self.beta(t)      
        gamma, beta = gamma.view(gamma.shape[0], 625, -1), beta.view(beta.shape[0], 625, -1)
        return gamma * x + beta

# ===================== Currently using this model ===================== #
class NMSLatentDisentangledDynamic(torch.nn.Module):
    def __init__(self, input_dims=80, hidden_dims=256, z_dims=64, n_component=2):
        super().__init__()

        self.conv_enc = nn.Conv2d(input_dims, hidden_dims, kernel_size=1)
        self.bilstm_enc = nn.LSTM(hidden_dims, hidden_dims // 2, num_layers=1, 
                                bidirectional=True, batch_first=True)

        self.mu_art, self.var_art = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.mu_dyn, self.var_dyn = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)

        self.bilstm = nn.LSTM(88 + z_dims * 2, hidden_dims // 2, num_layers=2, 
                                bidirectional=True, batch_first=True)
        self.out_conv = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)
        self.out_conv_2 = nn.Conv2d(hidden_dims, input_dims, kernel_size=1)

        self.n_component = n_component
        self.z_dims = z_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)
    
    def encode(self, x):
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        features = x
        features = torch.transpose(features, 1,2)
        features = features.unsqueeze(-1)
        q_zs = nn.ReLU()(self.conv_enc(features))
        q_zs = q_zs.squeeze(-1)
        q_zs = torch.transpose(q_zs, 1,2)
        q_zs = self.bilstm_enc(q_zs)[0]

        art_cls_lst = []
        dyn_cls_lst = []

        mu_art_lst, var_art_lst = self.mu_art(q_zs), self.var_art(q_zs).exp_()
        z_art_lst = repar(mu_art_lst, var_art_lst)

        mu_dyn_lst, var_dyn_lst = self.mu_dyn(q_zs), self.var_dyn(q_zs).exp_()
        z_dyn_lst = repar(mu_dyn_lst, var_dyn_lst)

        # change to dynamic
        for i in range(q_zs.shape[1]):
            _, cls_z_art_prob = self.approx_qy_x(z_art_lst[:, i, :], self.mu_art_lookup, 
                                                self.logvar_art_lookup, 
                                                n_component=self.n_component)
            art_cls_lst.append(cls_z_art_prob)

            _, cls_z_dyn_prob = self.approx_qy_x(z_dyn_lst[:, i, :], self.mu_dyn_lookup, 
                                                self.logvar_dyn_lookup, 
                                                n_component=self.n_component)
            dyn_cls_lst.append(cls_z_dyn_prob)
        
        art_cls_lst = torch.stack(art_cls_lst, dim=1)
        dyn_cls_lst = torch.stack(dyn_cls_lst, dim=1)
        
        # decoder
        z_lst = torch.cat([z_art_lst, z_dyn_lst], dim=-1)

        return z_lst, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst,\
                     z_dyn_lst, dyn_cls_lst, mu_dyn_lst, var_dyn_lst
    
    def decode(self, pr, z_lst):
        decoder_features = torch.cat([pr, z_lst], dim=-1)
        x_hat = self.bilstm(decoder_features)[0]

        x_hat = torch.transpose(x_hat, 1,2)
        x_hat = x_hat.unsqueeze(-1)
        x_hat = nn.ReLU()(self.out_conv(x_hat))
        x_hat = nn.Sigmoid()(self.out_conv_2(x_hat))
        x_hat = x_hat.squeeze(-1)
        x_hat = torch.transpose(x_hat, 1,2)

        return x_hat

    def forward(self, x, pr):

        z_lst, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst,\
                     z_dyn_lst, dyn_cls_lst, mu_dyn_lst, var_dyn_lst = self.encode(x)
        
        x_hat = self.decode(pr, z_lst)

        # return x_hat, z, cls_z_logits, cls_z_prob, Normal(mu, var)
        return x_hat, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst,\
                     z_dyn_lst, dyn_cls_lst, mu_dyn_lst, var_dyn_lst
    
    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=1.0)
        mu_lookup.weight.requires_grad = True
        self.mu_art_lookup = mu_lookup

        mu_lookup_2 = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup_2.weight, gain=1.0)
        mu_lookup_2.weight.requires_grad = True
        self.mu_dyn_lookup = mu_lookup_2

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_lookup = nn.Embedding(self.n_component, self.z_dims)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_art_lookup = logvar_lookup

        logvar_lookup_2 = nn.Embedding(self.n_component, self.z_dims)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup_2.weight, init_logvar)
        logvar_lookup_2.weight.requires_grad = logvar_trainable
        self.logvar_dyn_lookup = logvar_lookup_2

    def _infer_class(self, q_z):
        logLogit_qy_x, qy_x = self.approx_qy_x(q_z, self.mu_lookup, 
                                                self.logvar_lookup, 
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