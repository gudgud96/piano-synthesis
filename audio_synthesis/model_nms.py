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
        self.timbre_embedding = nn.Embedding(4, embed_dim)
        self.film1 = FiLM(embed_dim, 625 * hidden_dims)
        self.bilstm = nn.LSTM(hidden_dims, hidden_dims // 2, num_layers=1, 
                                bidirectional=True, batch_first=True)
        self.film2 = FiLM(embed_dim, 625 * hidden_dims)
        self.conv2 = nn.Conv2d(hidden_dims, 128, kernel_size=1)

    def forward(self, pr, t):
        x = torch.transpose(pr, 1,2)
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        t_embed = self.timbre_embedding(t)
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


