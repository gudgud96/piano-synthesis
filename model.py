from custom.layers import *
from custom.layers import Encoder

import sys
import torch
from torch import nn
import torch.distributions as dist
from torch.distributions import Normal
import random
import utils
import numpy as np

import torch
from functions import vq, vq_st
from tensorboardX import SummaryWriter
from progress.bar import Bar
import torch.nn.functional as F


class MusicTransformerVAE(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, 
                 filter_size=2048,
                 debug=False, loader_path=None, dist=False, writer=None):
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
            self.filter_size = filter_size

        self.writer = writer
        self.encoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq+1,
            filter_size=self.filter_size)
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x, length=None, writer=None):
        # if self.training or not self.infer:
        enc_src_mask, enc_tgt_mask, enc_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                                src=x, 
                                                                                trg=x)
        # encoder output
        enc_out, w = self.encoder(x, mask=enc_look_ahead_mask)
        enc_out = torch.mean(enc_out, dim=1).unsqueeze(1)      # mean aggregate global properties

        # decoder output
        x_padded = F.pad(input=x, pad=(1, 0, 0, 0), mode='constant', value=0)

        # new masks after aggregation
        dec_src_mask, dec_tgt_mask, dec_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq+1, 
                                                                                src=torch.argmax(enc_out, dim=-1), 
                                                                                trg=x_padded)

        dec_out, w = self.decoder(x_padded, dec_tgt_mask, dec_look_ahead_mask, self.training, enc_output=enc_out)
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


class GMMMusicTransformerVAE(MusicTransformerVAE):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, 
                 filter_size=2048,
                 n_component=4,
                 debug=False, loader_path=None, dist=False, writer=None):
        
        super().__init__(embedding_dim=embedding_dim, vocab_size=vocab_size, 
                        num_layer=num_layer, max_seq=max_seq, 
                        dropout=dropout, filter_size=filter_size)
        
        self.n_component = n_component

        self.latent_mu, self.latent_var = nn.Linear(embedding_dim, embedding_dim), \
                                            nn.Linear(embedding_dim, embedding_dim)
        
        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-3) 
    
    def forward(self, x):
        # if self.training or not self.infer:
        enc_src_mask, enc_tgt_mask, enc_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                                src=x, 
                                                                                trg=x)
        # encoder output
        enc_out, w = self.encoder(x, mask=enc_look_ahead_mask)
        enc_out = torch.mean(enc_out, dim=1)      # mean aggregate global properties

        enc_mu, enc_var = self.latent_mu(enc_out), self.latent_var(enc_out).exp_()
        dis = Normal(enc_mu, enc_var)

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z = repar(enc_mu, enc_var)

        # infer gaussian component
        logLogit_qy_x, qy_x = self.approx_qy_x(z, self.mu_lookup, self.logvar_lookup, n_component=self.n_component)
        _, y = torch.max(qy_x, dim=1)

        # decoder output
        x_padded = F.pad(input=x, pad=(1, 0, 0, 0), mode='constant', value=0)

        # new masks after aggregation
        dec_src_mask, dec_tgt_mask, dec_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq+1, 
                                                                                src=torch.argmax(z.unsqueeze(1), dim=-1), 
                                                                                trg=x_padded)

        dec_out, w = self.decoder(x_padded, dec_tgt_mask, dec_look_ahead_mask, self.training, enc_output=z.unsqueeze(1))
        dec_out = self.fc(dec_out)

        return dec_out, dis, z, logLogit_qy_x, qy_x, y

    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.embedding_dim)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=2)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_lookup = nn.Embedding(self.n_component, self.embedding_dim)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup

    def _infer_class(self, q_z):
        logLogit_qy_x, qy_x = self._approx_qy_x(q_z, self.mu_lookup, self.logvar_lookup, n_component=self.n_component)
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


class MelodyMusicTransformerVAE(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, 
                 filter_size=2048,
                 debug=False, loader_path=None, dist=False, writer=None):
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
            self.filter_size = filter_size

        self.writer = writer
        
        # performance encoder
        self.perf_encoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        
        # melody encoder
        self.mel_encoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)

        self.decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim * 2,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq+1,
            filter_size=self.filter_size)

        self.fc = torch.nn.Linear(self.embedding_dim * 2, self.vocab_size)

    def forward(self, perf, mel, length=None, writer=None):
        # encode
        perf_out = self.perf_encode(perf)
        mel_out = self.mel_encode(mel)
        enc_out = self.merge_mel_perf(mel_out, perf_out, mode="tile")

        # decoder output
        x_padded = F.pad(input=perf, pad=(1, 0, 0, 0), mode='constant', value=0)

        # new masks after aggregation
        dec_src_mask, dec_tgt_mask, dec_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq+1, 
                                                                                src=torch.argmax(enc_out, dim=-1), 
                                                                                trg=x_padded)

        dec_out, w = self.decoder(x_padded, dec_tgt_mask, dec_look_ahead_mask, self.training, enc_output=enc_out)
        dec_out = self.fc(dec_out)

        return dec_out
    
    def perf_encode(self, x):
        enc_src_mask, enc_tgt_mask, enc_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                                src=x, 
                                                                                trg=x)
        # encoder output
        enc_out, w = self.perf_encoder(x, mask=enc_look_ahead_mask)
        enc_out = torch.mean(enc_out, dim=1).unsqueeze(1)      # mean aggregate global properties
        return enc_out
    
    def mel_encode(self, x):
        enc_src_mask, enc_tgt_mask, enc_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                                src=x, 
                                                                                trg=x)
        # encoder output
        enc_out, w = self.mel_encoder(x, mask=enc_look_ahead_mask)
        return enc_out
    
    def merge_mel_perf(self, enc_mel, enc_perf, mode="tile"):        
        if mode == "sum":
            enc_mel += enc_perf
        elif mode == "cat":
            enc_mel = torch.cat([enc_mel, enc_perf], dim=1)     # time axis
        elif mode == "tile":
            enc_perf = torch.cat([enc_perf] * enc_mel.shape[1], dim=1)    # stack at time axis
            enc_mel = torch.cat([enc_mel, enc_perf], dim=-1)     # dimension axis
        
        return enc_mel


class MusicVCNet(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, 
                 filter_size=2048, mel_token_size=87+4,
                 n_component=4,
                 debug=False, loader_path=None, dist=False, writer=None):
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
            self.mel_token_size = mel_token_size
            self.dist = dist
            self.filter_size = filter_size
            self.n_component = n_component

        self.writer = writer
        
        # style encoder (style)
        self.style_encoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        
        # melody encoder (content)
        self.mel_encoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        
        # melody decoder
        self.mel_decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq+1,
            filter_size=self.filter_size)

        # global decoder
        self.global_decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim * 2,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq+1,
            filter_size=self.filter_size)

        self.instance_norm = nn.InstanceNorm1d(max_seq)
        self.instance_norm_2 = nn.InstanceNorm1d(max_seq)

        self.fc = torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_out = torch.nn.Linear(self.embedding_dim, self.vocab_size)
        self.fc_mel = torch.nn.Linear(self.embedding_dim, self.mel_token_size)
        self.gamma = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.beta = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.latent_mu = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.latent_var = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-3)

    def forward(self, perf, mel, length=None, writer=None):
        # encode
        _, _, enc_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                    src=perf, 
                                                                    trg=perf)
        # encoder output
        style_out, w = self.style_encoder(perf, mask=enc_look_ahead_mask)

        _, _, enc_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                    src=mel, 
                                                                    trg=mel)
        mel_out, w = self.mel_encoder(mel, mask=enc_look_ahead_mask)
        
        # for melody output, apply instance norm and mean aggregate
        mel_out = self.instance_norm(mel_out)
        mel_out = torch.mean(mel_out, dim=1)

        # for style output, mean aggregate and use GM-VAE
        style_out = torch.mean(style_out, dim=1)
        style_mu, style_var = self.latent_mu(style_out), self.latent_var(style_out).exp_()

        style_dis = Normal(style_mu, style_var)

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z_style = repar(style_mu, style_var)

        # infer gaussian component
        logLogit_qy_x, qy_x = self.approx_qy_x(z_style, self.mu_lookup, self.logvar_lookup, n_component=self.n_component)
        _, style_y = torch.max(qy_x, dim=1)

        # combine both parts for global decoding
        z_out = torch.cat([z_style, mel_out], dim=-1)

        # decoder output
        perf_padded = F.pad(input=perf, pad=(1, 0, 0, 0), mode='constant', value=0)
        mel_padded = F.pad(input=mel, pad=(1, 0, 0, 0), mode='constant', value=0)

        # decode melody first
        dec_src_mask, dec_tgt_mask, dec_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq+1, 
                                                                                src=torch.argmax(mel_out.unsqueeze(1), dim=-1), 
                                                                                trg=mel_padded)

        dec_mel_out, w = self.mel_decoder(mel_padded, dec_tgt_mask, dec_look_ahead_mask, self.training, enc_output=mel_out.unsqueeze(1))
        dec_mel_out = self.fc_mel(dec_mel_out)

        # decode globally
        dec_src_mask, dec_tgt_mask, dec_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq+1, 
                                                                                src=torch.argmax(z_out.unsqueeze(1), dim=-1), 
                                                                                trg=perf_padded)

        dec_global_out, w = self.global_decoder(perf_padded, dec_tgt_mask, dec_look_ahead_mask, self.training, enc_output=z_out.unsqueeze(1))
        dec_global_out = self.fc(dec_global_out)

        # apply adaptive instance normalization
        dec_global_out = self.instance_norm_2(dec_global_out)
        gamma, beta = self.gamma(z_style).unsqueeze(1), self.beta(z_style).unsqueeze(1)
        dec_global_out = dec_global_out * gamma + beta
        dec_global_out = self.fc_out(dec_global_out)

        return dec_global_out, dec_mel_out, style_dis, z_style, logLogit_qy_x, qy_x, style_y

    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.embedding_dim)
        nn.init.xavier_uniform_(mu_lookup.weight)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_lookup = nn.Embedding(self.n_component, self.embedding_dim)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup

    def _infer_class(self, q_z):
        logLogit_qy_x, qy_x = self._approx_qy_x(q_z, self.mu_lookup, self.logvar_lookup, n_component=self.n_component)
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


# ========================= VQ-VAE ========================== #
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        # self.embedding.weight.data.uniform_(-1./K, 1./K)
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.001)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.unsqueeze(1).unsqueeze(1)
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        z_q_x = z_q_x.squeeze(-1).squeeze(-1)

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()
        z_q_x_bar = z_q_x_bar.squeeze(-1).squeeze(-1)

        return z_q_x, z_q_x_bar, indices


class MusicVQVAE(nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, 
                 filter_size=2048, mel_token_size=87+4,
                 n_component=4,
                 debug=False, loader_path=None, dist=False, writer=None):
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
            self.mel_token_size = mel_token_size
            self.dist = dist
            self.filter_size = filter_size
            self.n_component = 4

        self.writer = writer
        
        # style encoder (style)
        self.style_encoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        
        self.codebook = VQEmbedding(self.n_component, self.embedding_dim)
        
        # melody encoder (content)
        self.mel_encoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        
        # melody decoder
        self.mel_decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq+1,
            filter_size=self.filter_size)

        # global decoder
        self.global_decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim * 2,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq+1,
            filter_size=self.filter_size)

        self.instance_norm = nn.InstanceNorm1d(max_seq)
        self.instance_norm_2 = nn.InstanceNorm1d(max_seq)

        self.fc = torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_out = torch.nn.Linear(self.embedding_dim, self.vocab_size)
        self.fc_mel = torch.nn.Linear(self.embedding_dim, self.mel_token_size)
        self.gamma = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.beta = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.latent_mu = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.latent_var = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
    
    def forward(self, perf, mel, length=None, writer=None):
        # encode
        _, _, enc_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                    src=perf, 
                                                                    trg=perf)
        # encoder output
        style_out, w = self.style_encoder(perf, mask=enc_look_ahead_mask)

        _, _, enc_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, 
                                                                    src=mel, 
                                                                    trg=mel)
        mel_out, w = self.mel_encoder(mel, mask=enc_look_ahead_mask)
        
        # for melody output, apply instance norm and mean aggregate
        mel_out = self.instance_norm(mel_out)
        mel_out = torch.mean(mel_out, dim=1)

        # for style output, mean aggregate and use VQ-VAE
        style_out = torch.mean(style_out, dim=1)
        z_q_style_st, z_q_style, style_y = self.codebook.straight_through(style_out)

        # combine both parts for global decoding
        z_out = torch.cat([z_q_style_st, mel_out], dim=-1)

        # decoder output
        perf_padded = F.pad(input=perf, pad=(1, 0, 0, 0), mode='constant', value=0)
        mel_padded = F.pad(input=mel, pad=(1, 0, 0, 0), mode='constant', value=0)

        # decode melody first
        dec_src_mask, dec_tgt_mask, dec_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq+1, 
                                                                                src=torch.argmax(mel_out.unsqueeze(1), dim=-1), 
                                                                                trg=mel_padded)

        dec_mel_out, w = self.mel_decoder(mel_padded, dec_tgt_mask, dec_look_ahead_mask, self.training, enc_output=mel_out.unsqueeze(1))
        dec_mel_out = self.fc_mel(dec_mel_out)

        # decode globally
        dec_src_mask, dec_tgt_mask, dec_look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq+1, 
                                                                                src=torch.argmax(z_out.unsqueeze(1), dim=-1), 
                                                                                trg=perf_padded)

        dec_global_out, w = self.global_decoder(perf_padded, dec_tgt_mask, dec_look_ahead_mask, self.training, enc_output=z_out.unsqueeze(1))
        dec_global_out = self.fc(dec_global_out)

        # apply adaptive instance normalization
        dec_global_out = self.instance_norm_2(dec_global_out)
        gamma, beta = self.gamma(z_q_style_st).unsqueeze(1), self.beta(z_q_style_st).unsqueeze(1)
        dec_global_out = dec_global_out * gamma + beta
        dec_global_out = self.fc_out(dec_global_out)

        return dec_global_out, dec_mel_out, style_out, z_q_style, style_y