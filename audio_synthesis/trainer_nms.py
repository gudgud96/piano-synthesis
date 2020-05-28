from dataset import MAESTRO
from torch.utils.data import Dataset, DataLoader
from model_nms import *
from nnAudio import Spectrogram
import torch
from torch.distributions import kl_divergence
from torch import optim
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
import json, os
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from collections import Counter


class Normalizer():
    """This class is for normalizing the spectrograms batch by batch. The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected. In this paper, we found that 'imagewise' normalization works better than 'framewise'"""
    def __init__(self, mode='framewise'):

        self.min_max_mean = None
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                output = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                output[torch.isnan(output)]=0 # Making nan to 0
                return output
        elif mode == 'imagewise':
            def normalize(x):
                size = x.shape
                x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                if self.min_max_mean is None:
                    self.min_max_mean = (x_min.mean(), x_max.mean())
                else:
                    self.min_max_mean = ((self.min_max_mean[0] + x_min.mean()) / 2, \
                                        (self.min_max_mean[1] + x_max.mean()) / 2)
                
                x_max = x_max.unsqueeze(1) # Make it broadcastable
                x_min = x_min.unsqueeze(1) # Make it broadcastable 
                return (x-x_min)/(x_max-x_min+1e-15)
        else:
            print('please choose the correct mode')
        self.normalize = normalize

    def transform(self, x):
        return self.normalize(x)
    
    def save_minmax(self):
        torch.save(self.min_max_mean, "normalizer.pt")


def loss_function(melspec_hat, melspec):
    recon_loss = torch.nn.MSELoss()(melspec_hat, melspec)
    return recon_loss


def training():
    step_unsup, step_sup = 0, 0
    learning_rate_counter = 0
    total_epoch = args['epochs']
    lr_factor = 2

    for ep in range(1, total_epoch):
        print("Epoch: {} / {}".format(ep, total_epoch))
        
        print("Unsupervised...")
        # train unsupervised
        for i, x in enumerate(train_dl):
            
            optimizer.zero_grad()

            audio, onset_pr, frame_pr = x     # (b, 320000), (b, t=625, 88)
            pr = torch.cat([onset_pr, frame_pr], dim=-1)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)

            # use log melspec
            pr = pr.cuda()
            if args["melspec_mode"] == "log":
                melspec = torch.log(melspec + 1e-12).cuda()
            elif args["melspec_mode"] == "log-tanh":
                melspec = torch.nn.Tanh()(0.25 * torch.log(melspec + 1e-12)).cuda()
            elif args["melspec_mode"] == "log-minmax":
                melspec = normalizer.transform(torch.log(melspec + 1e-12)).cuda()

            melspec_hat = model(pr, torch.zeros(pr.shape[0]).long().cuda())
            loss = loss_function(melspec_hat, melspec)
            
            loss.backward()
            optimizer.step()

            print("Batch {}/{}: Recon: {:.4}".format(i+1, len(train_dl), loss.item()), end="\r")

            train_unsup_writer.add_scalar('Recon', loss.item(), global_step=step_unsup)

            step_unsup += 1
            learning_rate_counter +=1
        
        # evaluate unsupervised
        eval_loss = 0
        for i, x in enumerate(val_dl):
            
            audio, onset_pr, frame_pr = x     # (b, 320000), (b, t=625, 88)
            pr = torch.cat([onset_pr, frame_pr], dim=-1)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)
            
            # use log melspec
            pr = pr.cuda()
            if args["melspec_mode"] == "log":
                melspec = torch.log(melspec + 1e-12).cuda()
            elif args["melspec_mode"] == "log-tanh":
                melspec = torch.nn.Tanh()(0.25 * torch.log(melspec + 1e-12)).cuda()
            elif args["melspec_mode"] == "log-minmax":
                melspec = normalizer.transform(torch.log(melspec + 1e-12)).cuda()
            melspec_hat = model(pr, torch.zeros(pr.shape[0]).long().cuda())
            loss = loss_function(melspec_hat, melspec)
            
            eval_loss += loss.item() / len(val_dl)

        print("Unsup Eval: Recon: {:.4}".format(eval_loss))
        eval_unsup_writer.add_scalar('Recon', eval_loss, global_step=step_unsup)

        # save model every epoch
        torch.save(model.state_dict(), save_path)
        normalizer.save_minmax()

        if step_unsup % 40000 == 0 and step_unsup > 0:
            for p in optimizer.param_groups:
                p['lr'] = args["lr"] / lr_factor
            lr_factor *= 2

        if ep % 1 == 0:
            # plot spectrograms
            audio, onset_pr, frame_pr = train_ds[10]
            pr_visualize = onset_pr + frame_pr
            pr = torch.cat([onset_pr, frame_pr], dim=-1)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)
            
            # use log melspec
            pr = pr.cuda().unsqueeze(0)
            if args["melspec_mode"] == "log":
                melspec = torch.log(melspec + 1e-12).cuda()
            elif args["melspec_mode"] == "log-tanh":
                melspec = torch.nn.Tanh()(0.25 * torch.log(melspec + 1e-12)).cuda()
            elif args["melspec_mode"] == "log-minmax":
                melspec = normalizer.transform(torch.log(melspec + 1e-12)).cuda()
            melspec_hat = model(pr, torch.zeros(pr.shape[0]).long().cuda())
    
            melspec_original = wav_to_melspec(audio).squeeze()
            
            if args["melspec_mode"] == "log":
                melspec_hat_denorm = torch.exp(melspec_hat).cuda().T.squeeze()
            elif args["melspec_mode"] == "log-tanh":
                def atanh(x):
                    return 0.5*torch.log((1+x)/(1-x) + 1e-12)
                melspec_hat_denorm = torch.exp(atanh(melspec_hat) * 4).cuda()
                print(melspec_hat_denorm.shape)
            elif args["melspec_mode"] == "log-minmax":
                melspec_hat_denorm = torch.exp(melspec_hat * (5.31 + 16.35) - 16.35).T.squeeze()

            # plot spectrograms
            fig = plt.figure(figsize=(8,8))
            melspec_db_1 = librosa.power_to_db(melspec_original.cpu().detach().numpy().squeeze(), ref=np.max)
            librosa.display.specshow(melspec_db_1, x_axis='time',
                                    y_axis='mel', sr=16000,
                                    fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            train_unsup_writer.add_figure('spec_original', fig, global_step=step_unsup, close=True)

            fig = plt.figure(figsize=(8,8))
            melspec_db_2 = librosa.power_to_db(melspec_hat_denorm.cpu().detach().numpy().squeeze(), ref=np.max)
            print(melspec_db_2.shape)
            librosa.display.specshow(melspec_db_2, x_axis='time',
                                    y_axis='mel', sr=16000,
                                    fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            train_unsup_writer.add_figure('spec_recon', fig, global_step=step_unsup, close=True)

            # plot piano rolls and features learnt
            fig = plt.figure(figsize=(8,8))
            plt.imshow(pr_visualize.squeeze().cpu().detach().numpy().T)
            train_unsup_writer.add_figure('onset_pr_original', fig, global_step=step_unsup, close=True)

if __name__ == "__main__":

    # housekeeping
    with open('nms_config.json') as f:
        args = json.load(f)
    if not os.path.isdir('params'):
        os.mkdir('params')
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    save_path = 'params/{}_{}.pt'.format(args['name'], datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    save_path += '_{}'.format(args["melspec_mode"])

    NUM_EMOTIONS = 4
    MELSPEC_DIM = 128
    PR_DIM = 88

    # load unlabelled data
    train_ds = MAESTRO(path='/data/MAESTRO', groups=['train_all'], sequence_length=320000)
    train_dl = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, num_workers=0)
    val_ds = MAESTRO(path='/data/MAESTRO', groups=['validation_all'], sequence_length=320000)
    val_dl = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)
    test_ds = MAESTRO(path='/data/MAESTRO', groups=['test_all'], sequence_length=320000)
    test_dl = DataLoader(test_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)

    # load model
    model = NMS()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), eps=1e-9)

    wav_to_melspec = Spectrogram.MelSpectrogram(sr=16000)
    normalizer = Normalizer(mode="imagewise")

    # load writers
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = 'logs/'+args['name']+'/'+current_time+'/train'
    eval_log_dir = 'logs/'+args['name']+'/'+current_time+'/eval'
    train_unsup_writer = SummaryWriter(train_log_dir + "_unsup")
    train_sup_writer = SummaryWriter(train_log_dir + "_sup")
    eval_unsup_writer = SummaryWriter(eval_log_dir + "_unsup")
    eval_sup_writer = SummaryWriter(eval_log_dir + "_sup")
    
    training()