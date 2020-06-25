import torch
from torch import nn
import numpy as np
import math as m
from torch.distributions import Normal
from torch.nn import functional as F
from layers import *
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



class NMS(torch.nn.Module):
    def __init__(self, embed_dim=2, input_dims=176, hidden_dims=256):
        super().__init__()

        self.conv1 = nn.Conv2d(88, hidden_dims, kernel_size=1)
        self.film1 = FiLM(embed_dim, 625 * hidden_dims)
        self.bilstm = nn.LSTM(hidden_dims, hidden_dims // 2, num_layers=1, 
                                bidirectional=True, batch_first=True)
        self.film2 = FiLM(embed_dim, 625 * hidden_dims)
        self.conv2 = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)

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

        self.conv_enc = nn.Conv2d(input_dims, hidden_dims, kernel_size=1)
        # self.conv_enc_2 = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)

        # self.lstm_art = nn.LSTM(hidden_dims, hidden_dims, num_layers=1, 
        #                         bidirectional=False, batch_first=True)
        # self.lstm_dyn = nn.LSTM(hidden_dims, hidden_dims, num_layers=1, 
        #                         bidirectional=False, batch_first=True)

        self.mu_art, self.var_art = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.mu_dyn, self.var_dyn = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)

        self.bilstm = nn.LSTM(88 + z_dims * 2, hidden_dims // 2, num_layers=2, 
                                bidirectional=True, batch_first=True)
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
        # q_z = nn.ReLU()(self.conv_enc_2(q_z))
        
        q_z = q_z.squeeze(-1)
        q_z = torch.transpose(q_z, 1,2)

        # q_z_art = self.lstm_art(q_z)[0][:, -1, :]
        # q_z_dyn = self.lstm_dyn(q_z)[0][:, -1, :]
        
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
        
        # cyclic
        x_hat_transpose = torch.transpose(x_hat, 1,2)
        x_hat_transpose = x_hat_transpose.unsqueeze(-1)
        q_z_2 = nn.ReLU()(self.conv_enc(x_hat_transpose))
        # q_z_2 = nn.ReLU()(self.conv_enc_2(q_z_2))
        
        q_z_2 = q_z_2.squeeze(-1)
        q_z_2 = torch.transpose(q_z_2, 1,2)

        # q_z_art_2 = self.lstm_art(q_z_2)[0][:, -1, :]
        # q_z_dyn_2 = self.lstm_dyn(q_z_2)[0][:, -1, :]

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
        self.out_conv_2 = nn.Conv2d(hidden_dims, 80, kernel_size=1)

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


class NMSLatentDisentangledStyle(torch.nn.Module):
    def __init__(self, input_dims=80, hidden_dims=256, z_dims=64, n_component=2, n_style=4):
        super().__init__()

        # encoder part
        self.conv_enc = nn.Conv2d(input_dims, hidden_dims, kernel_size=1)
        self.bilstm_enc = nn.LSTM(hidden_dims, hidden_dims // 2, num_layers=1, 
                                bidirectional=True, batch_first=True)

        self.mu_art, self.var_art = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.mu_dyn, self.var_dyn = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)

        # style encoding part
        self.style_lstm = nn.LSTM(z_dims * 2, z_dims // 2, num_layers=1, 
                                bidirectional=True, batch_first=True)
        self.mu_style, self.var_style = nn.Linear(z_dims, z_dims), nn.Linear(z_dims, z_dims)
        
        self.z_style_to_art, self.z_style_to_dyn = nn.Linear(z_dims, z_dims), nn.Linear(z_dims, z_dims)
        self.z_art_dec_cell = nn.GRUCell(2, z_dims)
        self.z_dyn_dec_cell = nn.GRUCell(2, z_dims)
        self.approx_art_prob, self.approx_dyn_prob = nn.Linear(z_dims, 2), nn.Linear(z_dims, 2)

        # decoder part
        self.bilstm = nn.LSTM(88 + z_dims * 2, hidden_dims // 2, num_layers=2, 
                                bidirectional=True, batch_first=True)
        self.out_conv = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)
        self.out_conv_2 = nn.Conv2d(hidden_dims, 80, kernel_size=1)

        self.n_component = n_component
        self.n_style = n_style
        self.z_dims = z_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)

    def forward(self, x, pr):
        
        # encoder
        z_lst, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst, \
                z_dyn_lst, dyn_cls_lst, mu_dyn_lst, var_dyn_lst = self.encode(x)
        
        # encode style
        z_style, z_style_dist, cls_z_style_logits, cls_z_style_prob = self.infer_style(z_lst)

        # decode z_lst
        art_cls_lst_hat, dyn_cls_lst_hat = self.decode_z_lst(z_style)     

        # decoder
        x_hat = self.decode(pr, z_lst)

        # return x_hat, z, cls_z_logits, cls_z_prob, Normal(mu, var)
        return x_hat, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst, \
                    z_dyn_lst, dyn_cls_lst, mu_dyn_lst, var_dyn_lst, \
                    z_style, z_style_dist, cls_z_style_logits, cls_z_style_prob, \
                    art_cls_lst_hat, dyn_cls_lst_hat

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

        z_lst = torch.cat([z_art_lst, z_dyn_lst], dim=-1)
        
        return z_lst, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst, \
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

    def infer_style(self, z_lst):
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        # use last latent state as style embedding
        q_z_style = self.style_lstm(z_lst)[0][:, -1, :]
        mu_style, var_style = self.mu_style(q_z_style), self.var_style(q_z_style)
        z_style = repar(mu_style, var_style)

        cls_z_style_logits, cls_z_style_prob = self.approx_qy_x(z_style, self.mu_style_lookup, 
                                                self.logvar_style_lookup, 
                                                n_component=self.n_style)

        return z_style, Normal(mu_style, var_style), cls_z_style_logits, cls_z_style_prob
    
    def decode_z_lst(self, z_style):
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        steps = 625
        z_art_hat, z_dyn_hat = self.z_style_to_art(z_style), self.z_style_to_dyn(z_style)
        hx_art = z_art_hat      # try without any transformation first
        hx_dyn = z_dyn_hat       # try without any transformation first
        x_in_art, x_in_dyn = torch.zeros(z_art_hat.shape[0], 2).cuda(), torch.zeros(z_dyn_hat.shape[0], 2).cuda()
        art_cls_lst, dyn_cls_lst = [], []

        for i in range(steps):
            hx_art = self.z_art_dec_cell(x_in_art, hx_art)
            x_out_art = self.approx_art_prob(hx_art)
            art_cls_lst.append(x_out_art)

            # max to one-hot and feed as input
            x_in_art = torch.zeros_like(x_out_art)
            arange = torch.arange(x_out_art.size(0)).long()
            x_in_art[arange, torch.argmax(x_out_art, dim=-1)] = 1
            
            hx_dyn = self.z_dyn_dec_cell(x_in_dyn, hx_dyn)
            x_out_dyn = self.approx_dyn_prob(hx_dyn)
            dyn_cls_lst.append(x_out_dyn)

            # max to one-hot and feed as input
            x_in_dyn = torch.zeros_like(x_out_dyn)
            arange = torch.arange(x_out_dyn.size(0)).long()
            x_in_dyn[arange, torch.argmax(x_out_dyn, dim=-1)] = 1

        art_cls_lst = torch.stack(art_cls_lst, dim=1)
        dyn_cls_lst = torch.stack(dyn_cls_lst, dim=1)

        return art_cls_lst, dyn_cls_lst

    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=1.0)
        mu_lookup.weight.requires_grad = True
        self.mu_art_lookup = mu_lookup

        mu_lookup_2 = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup_2.weight, gain=1.0)
        mu_lookup_2.weight.requires_grad = True
        self.mu_dyn_lookup = mu_lookup_2

        mu_lookup_3 = nn.Embedding(self.n_style, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup_3.weight, gain=1.0)
        mu_lookup_3.weight.requires_grad = True
        self.mu_style_lookup = mu_lookup_3

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

        logvar_lookup_3 = nn.Embedding(self.n_style, self.z_dims)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup_3.weight, init_logvar)
        logvar_lookup_3.weight.requires_grad = logvar_trainable
        self.logvar_style_lookup = logvar_lookup_3

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



class NMSLatentDisentangledStyleV2(torch.nn.Module):
    def __init__(self, input_dims=80, hidden_dims=256, z_dims=64, n_component=2, n_style=4):
        super().__init__()

        # encoder part
        self.conv_enc = nn.Conv2d(input_dims, hidden_dims, kernel_size=1)
        self.bilstm_enc = nn.LSTM(hidden_dims, hidden_dims // 2, num_layers=1, 
                                bidirectional=True, batch_first=True)

        self.mu_art, self.var_art = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.mu_dyn, self.var_dyn = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)

        # style encoding part
        self.style_lstm = nn.LSTM(z_dims * 2, z_dims // 2, num_layers=1, 
                                bidirectional=True, batch_first=True)
        self.mu_style, self.var_style = nn.Linear(z_dims, z_dims), nn.Linear(z_dims, z_dims)
        
        self.z_style_to_art, self.z_style_to_dyn = nn.Linear(z_dims, z_dims), nn.Linear(z_dims, z_dims)
        self.z_art_dec_cell = nn.GRUCell(z_dims, z_dims)
        self.z_dyn_dec_cell = nn.GRUCell(z_dims, z_dims)
        self.approx_art_prob, self.approx_dyn_prob = nn.Linear(z_dims, 2), nn.Linear(z_dims, 2)

        # decoder part
        self.bilstm = nn.LSTM(88 + z_dims * 2, hidden_dims // 2, num_layers=2, 
                                bidirectional=True, batch_first=True)
        self.out_conv = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=1)
        self.out_conv_2 = nn.Conv2d(hidden_dims, 80, kernel_size=1)

        self.n_component = n_component
        self.n_style = n_style
        self.z_dims = z_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)

    def forward(self, x, pr):
        
        # encoder
        z_lst, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst, \
                z_dyn_lst, dyn_cls_lst, mu_dyn_lst, var_dyn_lst = self.encode(x)
        
        # encode style
        z_style, z_style_dist, cls_z_style_logits, cls_z_style_prob = self.infer_style(z_lst)

        # decode z_lst
        z_art_lst_hat, z_dyn_lst_hat = self.decode_z_lst(z_style)     

        # decoder
        x_hat = self.decode(pr, z_lst)

        # return x_hat, z, cls_z_logits, cls_z_prob, Normal(mu, var)
        return x_hat, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst, \
                    z_dyn_lst, dyn_cls_lst, mu_dyn_lst, var_dyn_lst, \
                    z_style, z_style_dist, cls_z_style_logits, cls_z_style_prob, \
                    z_art_lst_hat, z_dyn_lst_hat

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

        z_lst = torch.cat([z_art_lst, z_dyn_lst], dim=-1)
        
        return z_lst, z_art_lst, art_cls_lst, mu_art_lst, var_art_lst, \
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

    def infer_style(self, z_lst):
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        # use last latent state as style embedding
        q_z_style = self.style_lstm(z_lst)[0][:, -1, :]
        mu_style, var_style = self.mu_style(q_z_style), self.var_style(q_z_style)
        z_style = repar(mu_style, var_style)

        cls_z_style_logits, cls_z_style_prob = self.approx_qy_x(z_style, self.mu_style_lookup, 
                                                self.logvar_style_lookup, 
                                                n_component=self.n_style)

        return z_style, Normal(mu_style, var_style), cls_z_style_logits, cls_z_style_prob
    
    def decode_z_lst(self, z_style):
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        steps = 625
        z_art_hat, z_dyn_hat = self.z_style_to_art(z_style), self.z_style_to_dyn(z_style)
        hx_art = nn.Tanh()(z_art_hat)      # try without any transformation first
        hx_dyn = nn.Tanh()(z_dyn_hat)      # try without any transformation first
        x_in_art, x_in_dyn = torch.zeros(z_art_hat.shape[0], z_art_hat.shape[1]).cuda(), torch.zeros(z_dyn_hat.shape[0], z_dyn_hat.shape[1]).cuda()
        z_art_lst, z_dyn_lst = [], []

        for i in range(steps):
            x_out_art = self.z_art_dec_cell(x_in_art, hx_art)
            z_art_lst.append(x_out_art)
            x_in_art, hx_art = x_out_art, nn.Tanh()(x_out_art)
            
            x_out_dyn = self.z_dyn_dec_cell(x_in_dyn, hx_dyn)
            z_dyn_lst.append(x_out_dyn)
            x_in_dyn, hx_dyn = x_out_dyn, nn.Tanh()(x_out_dyn)

        z_art_lst = torch.stack(z_art_lst, dim=1)
        z_dyn_lst = torch.stack(z_dyn_lst, dim=1)

        return z_art_lst, z_dyn_lst

    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=1.0)
        mu_lookup.weight.requires_grad = True
        self.mu_art_lookup = mu_lookup

        mu_lookup_2 = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup_2.weight, gain=1.0)
        mu_lookup_2.weight.requires_grad = True
        self.mu_dyn_lookup = mu_lookup_2

        mu_lookup_3 = nn.Embedding(self.n_style, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup_3.weight, gain=1.0)
        mu_lookup_3.weight.requires_grad = True
        self.mu_style_lookup = mu_lookup_3

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

        logvar_lookup_3 = nn.Embedding(self.n_style, self.z_dims)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup_3.weight, init_logvar)
        logvar_lookup_3.weight.requires_grad = logvar_trainable
        self.logvar_style_lookup = logvar_lookup_3

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


