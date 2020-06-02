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

        self.conv1 = nn.Conv2d(88, hidden_dims, kernel_size=1)
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
    def __init__(self, input_dims=80, hidden_dims=256, z_dims=64, n_component=2):
        super().__init__()

        # self.nms = NMS(embed_dim=z_dims)
        self.conv_enc = nn.Conv2d(input_dims, hidden_dims, kernel_size=1)
        self.conv_enc_2 = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)
        self.mu, self.var = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)

        self.bilstm = nn.LSTM(88 + z_dims + 1, hidden_dims, num_layers=2, 
                                bidirectional=False, batch_first=True)
        # self.out_linear = nn.Linear(hidden_dims, 128)
        self.out_conv = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)
        self.out_conv_2 = nn.Conv2d(hidden_dims, 80, kernel_size=1)

        self.n_component = n_component
        self.z_dims = z_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)

    def forward(self, x, pr, pedalling):

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
        q_z = nn.ReLU()(self.conv_enc(features))
        q_z = nn.ReLU()(self.conv_enc_2(q_z))
        
        q_z = q_z.squeeze(-1)
        q_z = torch.transpose(q_z, 1,2)
        
        q_z = torch.mean(q_z, dim=1)         # mean aggregate features

        mu, var = self.mu(q_z), self.var(q_z).exp_()
        z = repar(mu, var)
        cls_z_logits, cls_z_prob = self.approx_qy_x(z, self.mu_lookup, 
                                                    self.logvar_lookup, 
                                                    n_component=self.n_component)
        
        # decoder
        # x_hat = self.nms(pr, z)
        pedalling = pedalling.cuda().unsqueeze(-1).float()
        z = torch.cat([z, pedalling], dim=-1)
        z_distribute = torch.stack([z] * pr.shape[1], dim=1)
        decoder_features = torch.cat([pr, z_distribute], dim=-1)
        x_hat = self.bilstm(decoder_features)[0]

        x_hat = torch.transpose(x_hat, 1,2)
        x_hat = x_hat.unsqueeze(-1)
        x_hat = nn.ReLU()(self.out_conv(x_hat))
        x_hat = nn.Sigmoid()(self.out_conv_2(x_hat))
        x_hat = x_hat.squeeze(-1)
        x_hat = torch.transpose(x_hat, 1,2)

        # x_hat = nn.Sigmoid()(self.out_linear(x_hat))
        
        # cyclic
        x_hat_transpose = torch.transpose(x_hat, 1,2)
        x_hat_transpose = x_hat_transpose.unsqueeze(-1)
        q_z_2 = nn.ReLU()(self.conv_enc(x_hat_transpose))
        
        q_z_2 = q_z_2.squeeze(-1)
        q_z_2 = torch.transpose(q_z_2, 1,2)
        q_z_2 = torch.mean(q_z_2, dim=1)         # mean aggregate features
        mu_2, var_2 = self.mu(q_z_2), self.var(q_z_2).exp_()
        z_2 = repar(mu_2, var_2)
        cls_z_logits_2, cls_z_prob_2 = self.approx_qy_x(z_2, self.mu_lookup, 
                                                    self.logvar_lookup, 
                                                    n_component=self.n_component)


        # return x_hat, z, cls_z_logits, cls_z_prob, Normal(mu, var)
        return x_hat, z, cls_z_logits, cls_z_prob, Normal(mu, var), cls_z_prob_2
    
    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=1.0)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_lookup = nn.Embedding(self.n_component, self.z_dims)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
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



class NMSLatentDisentangled(torch.nn.Module):
    def __init__(self, input_dims=80, hidden_dims=256, z_dims=64, n_component=2):
        super().__init__()

        # self.nms = NMS(embed_dim=z_dims)
        self.conv_enc = nn.Conv2d(input_dims, hidden_dims, kernel_size=1)
        self.conv_enc_2 = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)
        self.mu_art, self.var_art = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.mu_dyn, self.var_dyn = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)

        self.bilstm = nn.LSTM(88 + z_dims * 2, hidden_dims // 2, num_layers=2, 
                                bidirectional=True, batch_first=True)
        # self.out_linear = nn.Linear(hidden_dims, 128)
        self.out_conv = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)
        self.out_conv_2 = nn.Conv2d(hidden_dims, 80, kernel_size=1)

        self.n_component = n_component
        self.z_dims = z_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)

    def forward(self, x, pr):

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        features = x
        features = torch.transpose(features, 1,2)
        features = features.unsqueeze(-1)
        q_z = nn.ReLU()(self.conv_enc(features))
        q_z = nn.ReLU()(self.conv_enc_2(q_z))
        
        q_z = q_z.squeeze(-1)
        q_z = torch.transpose(q_z, 1,2)
        
        q_z = torch.mean(q_z, dim=1)         # mean aggregate features

        mu_art, var_art = self.mu_art(q_z), self.var_art(q_z).exp_()
        z_art = repar(mu_art, var_art)
        cls_z_art_logits, cls_z_art_prob = self.approx_qy_x(z_art, self.mu_art_lookup, 
                                                            self.logvar_art_lookup, 
                                                            n_component=self.n_component)

        mu_dyn, var_dyn = self.mu_dyn(q_z), self.var_dyn(q_z).exp_()
        z_dyn = repar(mu_dyn, var_dyn)
        cls_z_dyn_logits, cls_z_dyn_prob = self.approx_qy_x(z_dyn, self.mu_dyn_lookup, 
                                                            self.logvar_dyn_lookup, 
                                                            n_component=self.n_component)
        
        # decoder
        # x_hat = self.nms(pr, z)
        z = torch.cat([z_art, z_dyn], dim=-1)
        z_distribute = torch.stack([z] * pr.shape[1], dim=1)
        decoder_features = torch.cat([pr, z_distribute], dim=-1)
        x_hat = self.bilstm(decoder_features)[0]

        x_hat = torch.transpose(x_hat, 1,2)
        x_hat = x_hat.unsqueeze(-1)
        x_hat = nn.ReLU()(self.out_conv(x_hat))
        x_hat = nn.Sigmoid()(self.out_conv_2(x_hat))
        x_hat = x_hat.squeeze(-1)
        x_hat = torch.transpose(x_hat, 1,2)

        # x_hat = nn.Sigmoid()(self.out_linear(x_hat))
        
        # cyclic
        x_hat_transpose = torch.transpose(x_hat, 1,2)
        x_hat_transpose = x_hat_transpose.unsqueeze(-1)
        q_z_2 = nn.ReLU()(self.conv_enc(x_hat_transpose))
        q_z_2 = nn.ReLU()(self.conv_enc_2(q_z_2))
        
        q_z_2 = q_z_2.squeeze(-1)
        q_z_2 = torch.transpose(q_z_2, 1,2)
        q_z_2 = torch.mean(q_z_2, dim=1)         # mean aggregate features
        mu_art_2, var_art_2 = self.mu_art(q_z_2), self.var_art(q_z_2).exp_()
        z_art_2 = repar(mu_art_2, var_art_2)
        cls_z_art_logits_2, cls_z_art_prob_2 = self.approx_qy_x(z_art_2, self.mu_art_lookup, 
                                                                self.logvar_art_lookup, 
                                                                n_component=self.n_component)
        
        mu_dyn_2, var_dyn_2 = self.mu_dyn(q_z_2), self.var_dyn(q_z_2).exp_()
        z_dyn_2 = repar(mu_dyn_2, var_dyn_2)
        cls_z_dyn_logits_2, cls_z_dyn_prob_2 = self.approx_qy_x(z_dyn_2, self.mu_dyn_lookup, 
                                                                self.logvar_dyn_lookup, 
                                                                n_component=self.n_component)


        # return x_hat, z, cls_z_logits, cls_z_prob, Normal(mu, var)
        return x_hat, z_art, z_dyn, cls_z_art_logits, cls_z_art_prob, Normal(mu_art, var_art), \
                cls_z_dyn_logits, cls_z_dyn_prob, Normal(mu_dyn, var_dyn), \
                cls_z_art_prob_2, cls_z_dyn_prob_2
    
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


