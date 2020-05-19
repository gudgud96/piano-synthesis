from ptb import *
from model import *
import json
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
import numpy as np
import time

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()


# ============ DATA PREPARATION ============ #
class CustomSchedule:
    def __init__(self, d_model, warmup_steps=4000, optimizer=None, name="noam"):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.name = name

        self._step = 0
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        
        if self.name == "noam":
            arg1 = step ** (-0.5)
            arg2 = step * (self.warmup_steps ** -1.5)
            return self.d_model ** (-0.5) * min(arg1, arg2)
        
        elif self.name == "rsqrt_decay":
            return 0.1 * (max(step, self.warmup_steps) ** (-0.5))
        
        else:
            print("Unknown optimizer name")

    def state_dict(self):
        return {
            "steps": self._step,
            "rate": self._rate
            }
    
    def load_state_dict(self, state_dict):
        self._step = state_dict["steps"]
        self._rate = state_dict["rate"]


# hyperparameters
with open('model_config.json') as f:
    args = json.load(f)
if not os.path.isdir('params'):
    os.mkdir('params')
if not os.path.isdir('logs'):
    os.mkdir('logs')
save_path = 'params/{}.pt'.format(args['name'])

# model
version = 2
model = MusicVQVAE(embedding_dim=args["hidden_dim"], 
                    vocab_size=388+2, 
                    mel_token_size=106,
                    num_layer=args["num_layer"],
                    max_seq=args["max_seq"], 
                    dropout=args["dropout"],
                    filter_size=args["filter_size"],
                    n_component=4)
# model = torch.load("params/mel-transformer-vae-{}.pt".format(version))
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(args["hidden_dim"], optimizer=optimizer, warmup_steps=8000,
                            name="noam")

# optimizer.load_state_dict(torch.load("params/opt-{}.pt".format(version)))
# scheduler.load_state_dict(torch.load("params/scheduler-{}.pt".format(version)))


# if torch.cuda.is_available():
#     print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
#     model.cuda()
# else:
#     print('CPU mode')

# multi-GPU set
# if torch.cuda.device_count() > 1:
#     single_model = model
#     model = torch.nn.DataParallel(model, output_device=torch.cuda.device_count()-1)

single_model = model

step, pre_epoch = 0, 0
batch_size = args["batch_size"]
model.train()

# dataloaders
is_shuffle = True
filenames = get_paths()
train_ds_dist = MaestroDataset(filenames, mode="train", is_mel=True)
train_dl_dist = DataLoader(train_ds_dist, batch_size=args["batch_size"], shuffle=is_shuffle, num_workers=0)
val_ds_dist = MaestroDataset(filenames, mode="val", is_mel=True)
val_dl_dist = DataLoader(val_ds_dist, batch_size=args["batch_size"], shuffle=False, num_workers=0)
test_ds_dist = MaestroDataset(filenames, mode="test", is_mel=True)
test_dl_dist = DataLoader(test_ds_dist, batch_size=args["batch_size"], shuffle=is_shuffle, num_workers=0)
print(len(train_ds_dist), len(val_ds_dist), len(test_ds_dist))

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+args['name']+'/'+current_time+'/train'
eval_log_dir = 'logs/'+args['name']+'/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)


def loss_function(dec_global_out, performance_tokens_padded,
                    dec_mel_out, mel_tokens_padded,
                    style_out, z_q_style,
                    step, beta=1):

    # anneal beta
    if step < 1000:
        beta0 = 0
    else:
        beta0 = min((step - 1000) / 10000 * beta, beta)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    perf_loss = criterion(dec_global_out.view(-1, dec_global_out.shape[-1]), 
                        performance_tokens_padded.view(-1))
    perf_acc = accuracy_score(performance_tokens_padded.view(-1).cpu().detach().numpy(),
                        torch.argmax(dec_global_out, dim=-1).view(-1).cpu().detach().numpy())
    
    mel_loss = criterion(dec_mel_out.view(-1, dec_mel_out.shape[-1]), 
                        mel_tokens_padded.view(-1))
    mel_acc = accuracy_score(mel_tokens_padded.view(-1).cpu().detach().numpy(),
                        torch.argmax(dec_mel_out, dim=-1).view(-1).cpu().detach().numpy())

    # Vector quantization objective
    loss_vq = F.mse_loss(z_q_style, style_out.detach())
    # Commitment objective
    loss_commit = F.mse_loss(style_out, z_q_style.detach())

    loss = perf_loss + mel_loss + beta0 * (loss_vq + loss_commit)
    return loss, perf_loss, perf_acc, mel_loss, mel_acc, loss_vq, loss_commit


def training_phase(model, optimizer, scheduler):

    step = 0
    
    for i in range(1, args['n_epochs'] + 1):
        print("Epoch {} / {}".format(i, args['n_epochs']))

        batch_loss = 0
        batch_acc = 0
        test_perf_loss = 0
        test_perf_acc = 0
        test_mel_loss = 0
        test_mel_acc = 0
        test_kld = 0

        len_dl = len(train_dl_dist)

        z_lst = []
        cluster_lst = []

        for j, x in enumerate(train_dl_dist):

            performance_tokens, mel_tokens = x
            performance_tokens = performance_tokens.cuda().long()
            mel_tokens = mel_tokens.cuda().long()

            optimizer.zero_grad()
            dec_global_out, dec_mel_out, style_out, z_q_style, style_y = model(performance_tokens, mel_tokens)

            z_lst.append(z_q_style.detach())
            cluster_lst.append(style_y.detach())

            performance_tokens_padded = F.pad(input=performance_tokens, 
                                              pad=(0, 1, 0, 0), mode='constant', value=1) 
            mel_tokens_padded = F.pad(input=mel_tokens, 
                                              pad=(0, 1, 0, 0), mode='constant', value=1) 
            
            loss, perf_loss, perf_acc, mel_loss, mel_acc, \
                loss_vq, loss_commit = loss_function(dec_global_out, performance_tokens_padded,
                                dec_mel_out, mel_tokens_padded,
                                style_out, z_q_style,
                                step, beta=args["beta"])

            print("Step: {} Perf: {:.4f} {:.4f} Mel: {:.4f} {:.4f} VQ: {:.4f} Commit: {:.4f}".format(
                            step, perf_loss.item(), perf_acc,
                            mel_loss.item(), mel_acc,
                            loss_vq.item(), loss_commit.item()), end="\r")
            
            loss.backward()
            scheduler.step()
        
            train_summary_writer.add_scalar('perf_loss', perf_loss.item(), global_step=step)
            train_summary_writer.add_scalar('perf_acc', perf_acc, global_step=step)
            train_summary_writer.add_scalar('mel_loss', mel_loss.item(), global_step=step)
            train_summary_writer.add_scalar('mel_acc', mel_acc, global_step=step)
            train_summary_writer.add_scalar('kld_loss', loss_vq.item() + loss_commit.item(), global_step=step)
            train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=step)

            if step % 250 == 0:

                print("Saving model", "params/music-vq-{}.pt".format(version))
                # for model, save the whole thing
                torch.save(single_model, "params/music-vq-{}.pt".format(version))
                torch.save(optimizer.state_dict(), "params/opt-music-vq-{}.pt".format(version))
                torch.save(scheduler.state_dict(), "params/scheduler-music-vq-{}.pt".format(version))
                
                print("Evaluation...")
                single_model.eval()

                # evaluate on vgmidi
                for j, x in tqdm(enumerate(val_dl_dist), total=len(val_dl_dist)):
                    
                    performance_tokens, mel_tokens = x
                    performance_tokens = performance_tokens.cuda().long()
                    mel_tokens = mel_tokens.cuda().long()

                    dec_global_out, dec_mel_out, style_out, z_q_style, style_y = model(performance_tokens, mel_tokens)

                    performance_tokens_padded = F.pad(input=performance_tokens, 
                                                    pad=(0, 1, 0, 0), mode='constant', value=1) 
                    mel_tokens_padded = F.pad(input=mel_tokens, 
                                                    pad=(0, 1, 0, 0), mode='constant', value=1) 

                    loss, perf_loss, perf_acc, mel_loss, mel_acc, \
                        loss_vq, loss_commit = loss_function(dec_global_out, performance_tokens_padded,
                                        dec_mel_out, mel_tokens_padded,
                                        style_out, z_q_style,
                                        step, beta=args["beta"])

                    test_perf_loss += perf_loss.item()
                    test_perf_acc += perf_acc
                    test_mel_loss += mel_loss.item()
                    test_mel_acc += mel_acc
                    test_kld += loss_vq.item() + loss_commit.item()
                
                eval_summary_writer.add_scalar('perf_loss', test_perf_loss / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('perf_acc', test_perf_acc / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('mel_loss', test_mel_loss / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('mel_acc', test_mel_acc / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('kld_loss', test_kld / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=step)

                print("Evaluation - Perf: {:.4f} {:.4f} Mel: {:.4f} {:.4f} VQ: {:.4f} Commit: {:.4f}".format(
                            step, perf_loss.item(), perf_acc,
                            mel_loss.item(), mel_acc,
                            loss_vq.item(), loss_commit.item()), end="\r")
                
                test_perf_loss = 0
                test_perf_acc = 0
                test_mel_loss = 0
                test_mel_acc = 0
                test_kld = 0
            
            step += 1
            model.train()
        
        # plot tsne after each epoch
        print("Plotting TSNE...")
        z_lst = torch.cat(z_lst, dim=0).cpu().detach()
        cluster_lst = torch.cat(cluster_lst, dim=0).cpu().numpy()
            
        tsne = TSNE(n_components=2, verbose=3)  #metric='manhattan'
        tsne_features = tsne.fit_transform(z_lst.squeeze())

        color = cluster_lst
        palette = sns.color_palette("bright", len(set(color)))
        fig = plt.figure(figsize=(8,8))
        sns.scatterplot(tsne_features[:,0], tsne_features[:,1], palette=palette, hue=color, legend='full')
        train_summary_writer.add_figure('tsne', fig, global_step=step, close=True)


def evaluation_phase(model, optimizer, scheduler):
    model.eval()

    def run(dl):
        test_loss = 0
        test_acc = 0

        for j, x in enumerate(dl):

            performance_tokens = x
            performance_tokens = performance_tokens.cuda().long()
    
            out = model(performance_tokens)

            performance_tokens_padded = F.pad(input=performance_tokens, 
                                            pad=(0, 1, 0, 0), mode='constant', value=1)

            criterion = nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(out.view(-1, out.shape[-1]), performance_tokens_padded.view(-1))  # autoencoder

            acc = accuracy_score(performance_tokens_padded.view(-1).cpu().detach().numpy(),
                                torch.argmax(out, dim=-1).view(-1).cpu().detach().numpy())
            
            print(loss.item(), acc)

            test_loss += loss.item()
            test_acc += acc
        
        print('evaluate loss: {:.5f}'.format(test_loss / len(dl)))
        print('evaluate acc: {:.5f}'.format(test_acc / len(dl)))
    
    # run(test_dl_dist)
    # run(train_dl_dist)
    run(val_dl_dist)


def testing(model, optimizer, scheduler):
    a = [i % 345 for i in range(2048)]
    a = torch.Tensor(a).cuda().unsqueeze(0).long()
    out = model(a)
    print(torch.argmax(out, dim=-1))



training_phase(model, optimizer, scheduler)
# evaluation_phase(model, optimizer, scheduler)
# testing(model, optimizer, scheduler)