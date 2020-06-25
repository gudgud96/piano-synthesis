from dataset import MAESTRO
from torch.utils.data import Dataset, DataLoader
from model import *
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

# housekeeping
with open('config.json') as f:
    args = json.load(f)
if not os.path.isdir('params'):
    os.mkdir('params')
if not os.path.isdir('logs'):
    os.mkdir('logs')
save_path = 'params/{}_{}.pt'.format(args['name'], datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

NUM_EMOTIONS = 4
MELSPEC_DIM = 128
PR_DIM = 88

# load unlabelled data
train_ds = MAESTRO(path='/data/MAESTRO', groups=['train'], sequence_length=320000)
train_dl = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, num_workers=0)
val_ds = MAESTRO(path='/data/MAESTRO', groups=['validation'], sequence_length=320000)
val_dl = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)
test_ds = MAESTRO(path='/data/MAESTRO', groups=['test'], sequence_length=320000)
test_dl = DataLoader(test_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)

# load emotion labelled data
emotion_train_ds = MAESTRO(path='/data/MAESTRO', groups=['train_emotion'], sequence_length=320000)
emotion_train_dl = DataLoader(emotion_train_ds, batch_size=args["batch_size"] // 2, shuffle=True, num_workers=0)
emotion_val_ds = MAESTRO(path='/data/MAESTRO', groups=['validation_emotion'], sequence_length=320000)
emotion_val_dl = DataLoader(emotion_val_ds, batch_size=args["batch_size"] // 2, shuffle=False, num_workers=0)
emotion_test_ds = MAESTRO(path='/data/MAESTRO', groups=['test_emotion'], sequence_length=320000)
emotion_test_dl = DataLoader(emotion_test_ds, batch_size=args["batch_size"] // 2, shuffle=False, num_workers=0)

# noam scheduler for training transformer
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


# load model
model = PianoTacotron(melspec_dim=MELSPEC_DIM, pr_dim=PR_DIM,
                        prenet_sizes=[256, 128], 
                        conv_dims=[32, 32, 64, 64, 128, 128],
                        lstm_dims=args["latent_enc_lstm_dims"], 
                        linear_dims=args["latent_enc_linear_dims"], 
                        kernel_size=args["latent_enc_conv_kernel_size"], 
                        stride=args["latent_enc_conv_stride"], 
                        t_num_layers=args["transformer_dec_layers"], 
                        t_dims=args["transformer_model_dim"], 
                        t_dropout=args["transformer_dropout"], 
                        t_maxlen=args["transformer_maxlen"],
                        z_dims=args["z_dims"], 
                        k_dims=args["num_components"], 
                        r=args["reduction_factor"])
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(args["transformer_model_dim"], optimizer=optimizer, warmup_steps=8000,
                            name="noam")
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


def loss_function(melspec_hat, melspec, z_s_prob, emotion_label, z_u_dist, 
                  z_s_dist, is_sup=False):
    
    def std_normal(shape):
        N = Normal(torch.zeros(shape), torch.ones(shape))
        if torch.cuda.is_available():
            N.loc = N.loc.cuda()
            N.scale = N.scale.cuda()
        return N

    recon_loss = torch.nn.MSELoss()(melspec_hat, melspec)
    normal = std_normal(z_u_dist.mean.size())
    kl_lat_unsup = kl_divergence(z_u_dist, normal).mean()
    kld_lat_sup = torch.Tensor([0]).cuda()

    for k in torch.arange(0, NUM_EMOTIONS):       # number of components
        # infer current p(z|y)
        mu_p_cls_zs, var_p_cls_zs = model.mu_zs_lookup(k.cuda()), model.logvar_zs_lookup(k.cuda()).exp_()
        dis_p_cls_zs = Normal(mu_p_cls_zs, var_p_cls_zs)
        kld_lat = torch.mean(kl_divergence(z_s_dist, dis_p_cls_zs), dim=-1)
        kld_lat *= z_s_prob[:, k]
        kld_lat_sup += kld_lat.mean()
    
    # KL class loss --> KL[q(y|x) || p(y)] = H(q(y|x)) - log p(y)
    def entropy(qy_x):
        return torch.mean(qy_x * torch.log(qy_x), dim=1)

    kld_cls = (entropy(z_s_prob) - np.log(1 / NUM_EMOTIONS)).mean()
    
    if not is_sup:
        loss = recon_loss + kl_lat_unsup + kld_lat_sup + kld_cls
        return loss, recon_loss, kl_lat_unsup, kld_lat_sup, kld_cls

    else:
        mu_p_cls_zs, var_p_cls_zs = model.mu_zs_lookup(emotion_label.cuda().long()), \
                                    model.logvar_zs_lookup(emotion_label.cuda().long()).exp_()
        dis_p_cls_zs = Normal(mu_p_cls_zs, var_p_cls_zs)
        kld_lat_sup = torch.mean(kl_divergence(z_s_dist, dis_p_cls_zs), dim=-1).mean()

        clf_loss = torch.nn.CrossEntropyLoss()(z_s_prob, emotion_label)
        clf_acc = accuracy_score(torch.argmax(z_s_prob, dim=-1).cpu().detach().numpy(),
                                 emotion_label.cpu().detach().numpy())
        loss = recon_loss + kl_lat_unsup + kld_lat_sup + clf_loss
        return loss, recon_loss, kl_lat_unsup, kld_lat_sup, clf_loss, clf_acc


def training():
    step_unsup, step_sup = 0, 0
    learning_rate_counter = 0
    total_epoch = args['epochs']

    for ep in range(1, total_epoch):
        print("Epoch: {} / {}".format(ep, total_epoch))
        
        print("Unsupervised...")
        # train unsupervised
        for i, x in enumerate(train_dl):
            
            optimizer.zero_grad()

            audio, onset_pr = x     # (b, 320000), (b, t=625, 88)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)
            melspec, onset_pr = normalizer.transform(torch.log(melspec + 1e-12)).cuda(), onset_pr.cuda()
            melspec_hat, z_s_prob, z_u_dist, z_s_dist = model(melspec, onset_pr, is_sup=False, z_s=None)

            loss, recon_loss, kl_lat_unsup, kld_lat_sup, kld_cls = loss_function(melspec_hat, melspec, z_s_prob, 
                                                                                None, z_u_dist, z_s_dist,
                                                                                is_sup=False)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scheduler.step()

            print("Batch {}/{}: Recon: {:.4} KL Unsup: {:.4} KL Sup: {:.4} KL Cls: {:.4}".format(i+1, len(train_dl),
                                                                                                recon_loss.item(), 
                                                                                                kl_lat_unsup.item(),
                                                                                                kld_lat_sup.item(),
                                                                                                kld_cls.item()),
                                                                                                end="\r")

            train_unsup_writer.add_scalar('Recon', recon_loss.item(), global_step=step_unsup)
            train_unsup_writer.add_scalar('KL Unsup', kl_lat_unsup.item(), global_step=step_unsup)
            train_unsup_writer.add_scalar('KL Sup', kld_lat_sup.item(), global_step=step_unsup)
            train_unsup_writer.add_scalar('KL Cls', kld_cls.item(), global_step=step_unsup)
            train_unsup_writer.add_scalar('learning_rate', scheduler.optimizer.param_groups[0]["lr"], 
                                            global_step=learning_rate_counter)

            step_unsup += 1
            learning_rate_counter +=1
        
        # evaluate unsupervised
        eval_loss, eval_recon_loss, eval_kl_unsup_loss, eval_kl_sup_loss, eval_kl_cls = 0, 0, 0, 0, 0
        for i, x in enumerate(val_dl):
            
            audio, onset_pr = x     # (b, 320000), (b, t=625, 88)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)
            melspec, onset_pr = normalizer.transform(torch.log(melspec + 1e-12)).cuda(), onset_pr.cuda()
            melspec_hat, z_s_prob, z_u_dist, z_s_dist = model(melspec, onset_pr, is_sup=False, z_s=None)

            loss, recon_loss, kl_lat_unsup, kld_lat_sup, kld_cls = loss_function(melspec_hat, melspec, z_s_prob, 
                                                                                None, z_u_dist, z_s_dist,
                                                                                is_sup=False)
            
            eval_loss += loss.item() / len(val_dl)
            eval_recon_loss += recon_loss.item() / len(val_dl)
            eval_kl_unsup_loss += kl_lat_unsup.item() / len(val_dl)
            eval_kl_sup_loss += kld_lat_sup.item() / len(val_dl)
            eval_kl_cls += kld_cls.item() / len(val_dl)

        print("Unsup Eval: Recon: {:.4} KL Unsup: {:.4} KL Sup: {:.4} KL Cls: {:.4}".format(eval_recon_loss, 
                                                                                            eval_kl_unsup_loss,
                                                                                            eval_kl_sup_loss, 
                                                                                            eval_kl_cls))
        eval_unsup_writer.add_scalar('Recon', eval_recon_loss, global_step=step_unsup)
        eval_unsup_writer.add_scalar('KL Unsup', eval_kl_unsup_loss, global_step=step_unsup)
        eval_unsup_writer.add_scalar('KL Sup', eval_kl_sup_loss, global_step=step_unsup)
        eval_unsup_writer.add_scalar('KL Cls', eval_kl_cls, global_step=step_unsup)


        print("Supervised...")
        # train supervised
        for i, x in enumerate(emotion_train_dl):

            optimizer.zero_grad()

            audio, onset_pr, emotion_label = x   # (b, 320000), (b, t=625, 88), (b,)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)

            # convert emotion labels to one-hot
            emotion_label_oh = torch.zeros(emotion_label.shape[0], NUM_EMOTIONS)
            emotion_label_oh[torch.arange(emotion_label.shape[0]), emotion_label] = 1

            melspec, onset_pr, emotion_label, emotion_label_oh \
                = normalizer.transform(torch.log(melspec + 1e-12)).cuda(), onset_pr.cuda(), emotion_label.cuda(), emotion_label_oh.cuda()
            melspec_hat, z_s_prob, z_u_dist, z_s_dist = model(melspec, onset_pr, is_sup=True, z_s=emotion_label)

            loss, recon_loss, kl_lat_unsup, kl_lat_sup, clf_loss, clf_acc = loss_function(melspec_hat, melspec, z_s_prob, 
                                                                                        emotion_label, z_u_dist, z_s_dist,
                                                                                        is_sup=True)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scheduler.step()

            print("Batch {}/{}: Recon: {:.4} KL Unsup: {:.4} KL Sup: {:.4} CLF Loss: {:.4} Acc: {:.4}".format(i+1, 
                                                                                                            len(emotion_train_dl),
                                                                                                            recon_loss.item(), 
                                                                                                            kl_lat_unsup.item(), 
                                                                                                            kl_lat_sup.item(),
                                                                                                            clf_loss.item(), 
                                                                                                            clf_acc), end="\r")
                                            
            train_sup_writer.add_scalar('Recon', recon_loss.item(), global_step=step_sup)
            train_sup_writer.add_scalar('KL Unsup', kl_lat_unsup.item(), global_step=step_sup)
            train_sup_writer.add_scalar('KL Sup', kl_lat_sup, global_step=step_unsup)
            train_sup_writer.add_scalar('CLF Loss', clf_loss.item(), global_step=step_sup)
            train_sup_writer.add_scalar('CLF Acc', clf_acc, global_step=step_sup)
            train_unsup_writer.add_scalar('learning_rate', scheduler.optimizer.param_groups[0]["lr"], 
                                            global_step=learning_rate_counter)

            step_sup += 1
            learning_rate_counter += 1
            
        # evaluate supervised
        eval_loss, eval_recon_loss, eval_kl_unsup_loss, eval_kl_sup_loss, eval_clf_loss, eval_clf_acc = 0, 0, 0, 0, 0, 0
        for i, x in enumerate(emotion_val_dl):

            audio, onset_pr, emotion_label = x   # (b, 320000), (b, t=625, 88), (b,)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)

            # convert emotion labels to one-hot
            emotion_label_oh = torch.zeros(emotion_label.shape[0], NUM_EMOTIONS)
            emotion_label_oh[torch.arange(emotion_label.shape[0]), emotion_label] = 1

            melspec, onset_pr, emotion_label, emotion_label_oh \
                = normalizer.transform(torch.log(melspec + 1e-12)).cuda(), onset_pr.cuda(), emotion_label.cuda(), emotion_label_oh.cuda()
            melspec_hat, z_s_prob, z_u_dist, z_s_dist = model(melspec, onset_pr, is_sup=True, z_s=emotion_label)

            loss, recon_loss, kl_lat_unsup, kl_lat_sup, clf_loss, clf_acc = loss_function(melspec_hat, melspec, z_s_prob, 
                                                                                        emotion_label, z_u_dist, z_s_dist,
                                                                                        is_sup=True)

            eval_loss += loss.item() / len(emotion_val_dl)
            eval_recon_loss += recon_loss.item() / len(emotion_val_dl)
            eval_kl_unsup_loss += kl_lat_unsup.item() / len(emotion_val_dl)
            eval_kl_sup_loss += kl_lat_sup.item() / len(emotion_val_dl)
            eval_clf_loss += clf_loss.item() / len(emotion_val_dl)
            eval_clf_acc += clf_acc / len(emotion_val_dl)
            
        print("Sup Eval: Recon: {:.4} KL Unsup: {:.4} KL Sup: {:.4} Clf Loss: {:.4} Acc: {:.4}".format(eval_recon_loss, 
                                                                                                    eval_kl_unsup_loss,
                                                                                                    eval_kl_sup_loss,
                                                                                                    eval_clf_loss,
                                                                                                    eval_clf_acc))

        eval_sup_writer.add_scalar('Recon', eval_recon_loss, global_step=step_sup)
        eval_sup_writer.add_scalar('KL Unsup', eval_kl_unsup_loss, global_step=step_sup)
        eval_sup_writer.add_scalar('KL Sup', eval_kl_sup_loss, global_step=step_unsup)
        eval_sup_writer.add_scalar('CLF Loss', eval_clf_loss, global_step=step_sup)
        eval_sup_writer.add_scalar('CLF Acc', eval_clf_acc, global_step=step_sup)


        # save model every epoch
        torch.save(model.state_dict(), save_path)

training()