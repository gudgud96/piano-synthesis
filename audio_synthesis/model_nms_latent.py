import torch
from torch import nn
import numpy as np
import math as m
from torch.distributions import Normal
from torch.nn import functional as F
from layers import *
from tqdm import tqdm


class FiLM(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.gamma = nn.Linear(input_dims, output_dims)
        self.beta = nn.Linear(input_dims, output_dims)
    
    def forward(self, x, t):
        gamma, beta = self.gamma(t), self.beta(t)      
        gamma, beta = gamma.view(gamma.shape[0], 625, -1), beta.view(beta.shape[0], 625, -1)
        return gamma * x + beta



class NMS(torch.nn.Module):
    def __init__(self, embed_dim=2, input_dims=176, hidden_dims=256):
        super().__init__()

        self.conv1 = nn.Conv2d(176, hidden_dims, kernel_size=1)
        self.film1 = FiLM(embed_dim, 625 * hidden_dims)
        self.bilstm = nn.LSTM(hidden_dims, hidden_dims // 2, num_layers=1, 
                                bidirectional=True, batch_first=True)
        self.film2 = FiLM(embed_dim, 625 * hidden_dims)
        self.conv2 = nn.Conv2d(hidden_dims, 128, kernel_size=1)

    def forward(self, pr, t_embed):
        x = torch.transpose(pr, 1,2)
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        x = x.squeeze(-1)
        x = torch.transpose(x, 1,2)
        x = self.film1(x, t_embed)
        x = self.bilstm(x)[0]
        x = self.film2(x, t_embed)
        
        x = torch.transpose(x, 1,2)
        x = x.unsqueeze(-1)
        x = self.conv2(x)
        x = x.squeeze(-1)
        x = torch.transpose(x, 1,2)

        return x


class NMSLatent(torch.nn.Module):
    def __init__(self, input_dims=128, hidden_dims=256, z_dims=64, n_component=2):
        super().__init__()

        self.nms = NMS(embed_dim=z_dims)
        self.conv_enc = nn.Conv2d(input_dims, hidden_dims, kernel_size=1)
        self.conv_enc2 = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)
        self.mu, self.var = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)

        self.n_component = n_component
        self.z_dims = z_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)

    def forward(self, x, pr):

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        # features = torch.cat([x, pr], dim=-1)
        # features = torch.transpose(features, 1,2)
        # features = features.unsqueeze(-1)
        # q_z = self.conv_enc(features)
        features = x
        features = torch.transpose(features, 1,2)
        features = features.unsqueeze(-1)
        features = self.conv_enc(features)
        q_z = self.conv_enc2(features)
        
        q_z = q_z.squeeze(-1)
        q_z = torch.transpose(q_z, 1,2)
        
        q_z = torch.mean(q_z, dim=1)         # mean aggregate features

        mu, var = self.mu(q_z), self.var(q_z).exp_()
        z = repar(mu, var)
        cls_z_logits, cls_z_prob = self.approx_qy_x(z, self.mu_lookup, 
                                                    self.logvar_lookup, 
                                                    n_component=self.n_component)
        x_hat = self.nms(pr, z)

        return x_hat, z, cls_z_logits, cls_z_prob, Normal(mu, var)
    
    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=2.0)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_lookup = nn.Embedding(self.n_component, self.z_dims)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        print("logvar", np.isnan(init_logvar), np.isinf(init_logvar))
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup

    def _infer_class(self, q_z):
        logLogit_qy_x, qy_x = self._approx_qy_x(q_z, self.mu_lookup, 
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


class NMSLatentContinuous(torch.nn.Module):
    def __init__(self, input_dims=304, hidden_dims=256, z_dims=64, n_component=2):
        super().__init__()

        self.nms = NMS(embed_dim=z_dims)
        self.conv_enc = nn.Conv2d(input_dims, hidden_dims, kernel_size=1)
        self.mu_qz_x, self.var_qz_x = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.mu_qy_z, self.var_qy_z = nn.Linear(z_dims, 2), nn.Linear(z_dims, 2)
        self.mu_pz_y, self.var_pz_y = nn.Linear(2, z_dims), nn.Linear(2, z_dims)

        self.n_component = n_component
        self.z_dims = z_dims

    def forward(self, x, pr, y=None):

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        features = torch.cat([x, pr], dim=-1)
        features = torch.transpose(features, 1,2)
        features = features.unsqueeze(-1)
        q_z = self.conv_enc(features)
        q_z = q_z.squeeze(-1)
        q_z = torch.transpose(q_z, 1,2)
        
        q_z = torch.mean(q_z, dim=1)         # mean aggregate features

        mu, var = self.mu_qz_x(q_z), self.var_qz_x(q_z).exp_()
        z = repar(mu, var)
        y_mu, y_var = self.mu_qy_z(z), self.var_qy_z(z).exp_()
        
        y_pred = repar(y_mu, y_var)
        y_pred = nn.Sigmoid()(y_pred)

        if y is None:
            z_theta_mu, z_theta_var = self.mu_pz_y(y_pred), self.var_pz_y(y_pred).exp_()
        else:
            z_theta_mu, z_theta_var = self.mu_pz_y(y), self.var_pz_y(y).exp_()
        z_theta = repar(z_theta_mu, z_theta_var)

        x_hat = self.nms(pr, z_theta)

        return x_hat, z, y_pred, Normal(mu, var), Normal(z_theta_mu, z_theta_var), Normal(y_mu, y_var)


