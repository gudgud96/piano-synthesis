from dataset import MAESTRO
from torch.utils.data import Dataset, DataLoader
from model_nms_latent import *
from nnAudio import Spectrogram
import torch
from torch.distributions import kl_divergence, Normal
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
                # x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                # x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                # fix constant min max to be used
                x_max = torch.Tensor([[10]]).cuda()
                x_min = torch.Tensor([[-20]]).cuda()

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


def loss_function(melspec_hat, melspec, cls_z_prob, z_dist, is_sup=False, emotion_cls=None,
                 step=None, beta=1):
    
    # kl annealing
    if not step is None:
        if step < 1000:
            beta_0 = 0
        else:
            beta_0 = min((step - 1000) / 5000 * beta, beta)
    else:
        beta_0 = 1
    
    recon_loss = torch.nn.MSELoss()(melspec_hat, melspec)

    if is_sup:
        # weights = [0.2, 1.0]
        # class_weights = torch.FloatTensor(weights).cuda()
        cls_loss = torch.nn.CrossEntropyLoss()(cls_z_prob, emotion_cls)
        
        mu, var = model.mu_lookup(emotion_cls.cuda().long()), \
                        model.logvar_lookup(emotion_cls.cuda().long()).exp_()
        dis = Normal(mu, var)
        kl_loss = kl_divergence(z_dist, dis).mean()
        
        clf_acc = accuracy_score(torch.argmax(cls_z_prob, dim=-1).cpu().detach().numpy(),
                                    emotion_cls.cpu().detach().numpy())

        loss = recon_loss + cls_loss + beta_0 * kl_loss
        return loss, recon_loss, cls_loss, kl_loss, clf_acc
    
    else:
        kl_loss = 0
        for k in torch.arange(0, NUM_EMOTIONS):       # number of components
            # infer current p(z|y)
            mu, var = model.mu_lookup(k.cuda()), model.logvar_lookup(k.cuda()).exp_()
            dis = Normal(mu, var)
            kld_lat = torch.mean(kl_divergence(z_dist, dis), dim=-1)
            kld_lat *= cls_z_prob[:, k]
            kl_loss += kld_lat.mean()
        
        # KL class loss --> KL[q(y|x) || p(y)] = H(q(y|x)) - log p(y)
        def entropy(qy_x):
            return torch.mean(qy_x * torch.log(qy_x), dim=1)

        kl_cls = (entropy(cls_z_prob) - np.log(1 / NUM_EMOTIONS)).mean()

        loss = recon_loss + beta_0 * (kl_loss + kl_cls)
        return loss, recon_loss, kl_cls, kl_loss, None


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

            audio, onset_pr, frame_pr, _ = x     # (b, 320000), (b, t=625, 88)
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

            melspec_hat, z, cls_z_logits, cls_z_prob, z_dist = model(melspec, pr)
            loss, recon_loss, cls_loss, kl_loss, clf_acc = loss_function(melspec_hat, melspec, cls_z_prob, z_dist)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            print("Batch {}/{}: Recon: {:.4} KL Sup: {:.4} KL Loss: {:.4}".format(i+1, 
                                                                                len(train_dl),
                                                                                recon_loss.item(), 
                                                                                kl_loss.item(),
                                                                                cls_loss.item()), 
                                                                                end="\r")
                                            
            train_unsup_writer.add_scalar('Recon', recon_loss.item(), global_step=step_sup)
            train_unsup_writer.add_scalar('KL Unsup', kl_loss.item(), global_step=step_unsup)
            train_unsup_writer.add_scalar('KL CLF', cls_loss.item(), global_step=step_sup)

            step_unsup += 1
            learning_rate_counter +=1
        
        # evaluate unsupervised
        eval_loss, eval_recon_loss, eval_cls_loss, eval_cls_acc, eval_kl = 0, 0, 0, 0, 0
        for i, x in enumerate(val_dl):
            
            audio, onset_pr, frame_pr, _ = x     # (b, 320000), (b, t=625, 88)
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
            
            melspec_hat, z, cls_z_logits, cls_z_prob, z_dist = model(melspec, pr)
            loss, recon_loss, cls_loss, kl_loss, clf_acc = loss_function(melspec_hat, melspec, cls_z_prob, z_dist)
            
            eval_loss += loss.item() / len(val_dl)
            eval_recon_loss += recon_loss.item() / len(val_dl)
            eval_cls_loss += cls_loss.item() / len(val_dl)
            eval_kl = kl_loss.item() / len(val_dl)

        print("Unsup Eval: Recon: {:.4} KL Sup: {:.4} KL CLF: {:.4}".format(eval_recon_loss, 
                                                                            eval_kl,
                                                                            eval_cls_loss))
        
        eval_unsup_writer.add_scalar('Recon', eval_recon_loss, global_step=step_unsup)
        eval_unsup_writer.add_scalar('KL Unsup', eval_kl, global_step=step_unsup)
        eval_unsup_writer.add_scalar('KL CLF', eval_cls_loss, global_step=step_sup)

        print("Supervised...")
        # train supervised
        for i, x in enumerate(train_emotion_dl):
            
            optimizer.zero_grad()

            audio, onset_pr, frame_pr, _, emotion_cls = x     # (b, 320000), (b, t=625, 88)
            pr = torch.cat([onset_pr, frame_pr], dim=-1)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)

            # use log melspec
            pr = pr.cuda()
            emotion_cls = emotion_cls.cuda()
            if args["melspec_mode"] == "log":
                melspec = torch.log(melspec + 1e-12).cuda()
            elif args["melspec_mode"] == "log-tanh":
                melspec = torch.nn.Tanh()(0.25 * torch.log(melspec + 1e-12)).cuda()
            elif args["melspec_mode"] == "log-minmax":
                melspec = normalizer.transform(torch.log(melspec + 1e-12)).cuda()

            melspec_hat, z, cls_z_logits, cls_z_prob, z_dist = model(melspec, pr)
            loss, recon_loss, cls_loss, kl_loss, clf_acc = loss_function(melspec_hat, melspec, cls_z_prob, z_dist,
                                                                        is_sup=True, emotion_cls=emotion_cls)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            print("Batch {}/{}: Recon: {:.4} KL Sup: {:.4} CLF Loss: {:.4} Acc: {:.4}".format(i+1, 
                                                                                        len(train_dl),
                                                                                        recon_loss.item(), 
                                                                                        kl_loss.item(),
                                                                                        cls_loss.item(), 
                                                                                        clf_acc), end="\r")
                                            
            train_sup_writer.add_scalar('Recon', recon_loss.item(), global_step=step_sup)
            train_sup_writer.add_scalar('KL Sup', kl_loss.item(), global_step=step_sup)
            train_sup_writer.add_scalar('CLF Loss', cls_loss.item(), global_step=step_sup)
            train_sup_writer.add_scalar('CLF Acc', clf_acc, global_step=step_sup)

            step_sup += 1
            learning_rate_counter +=1
        
        # evaluate supervised
        eval_loss, eval_recon_loss, eval_cls_loss, eval_cls_acc, eval_kl = 0, 0, 0, 0, 0
        for i, x in enumerate(val_emotion_dl):
            
            audio, onset_pr, frame_pr, _, emotion_cls = x     # (b, 320000), (b, t=625, 88)
            pr = torch.cat([onset_pr, frame_pr], dim=-1)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)
            
            # use log melspec
            pr = pr.cuda()
            emotion_cls = emotion_cls.cuda()
            if args["melspec_mode"] == "log":
                melspec = torch.log(melspec + 1e-12).cuda()
            elif args["melspec_mode"] == "log-tanh":
                melspec = torch.nn.Tanh()(0.25 * torch.log(melspec + 1e-12)).cuda()
            elif args["melspec_mode"] == "log-minmax":
                melspec = normalizer.transform(torch.log(melspec + 1e-12)).cuda()
            
            melspec_hat, z, cls_z_logits, cls_z_prob, z_dist = model(melspec, pr)
            loss, recon_loss, cls_loss, kl_loss, clf_acc = loss_function(melspec_hat, melspec, cls_z_prob, z_dist,
                                                                    is_sup=True, emotion_cls=emotion_cls)
            
            eval_loss += loss.item() / len(val_emotion_dl)
            eval_recon_loss += recon_loss.item() / len(val_emotion_dl)
            eval_cls_loss += cls_loss.item() / len(val_emotion_dl)
            eval_cls_acc += clf_acc / len(val_emotion_dl)
            eval_kl = kl_loss.item() / len(val_emotion_dl)

        print("Sup Eval: Recon: {:.4} KL Sup: {:.4} CLF Loss: {:.4} Acc: {:.4}".format(
                                                                                eval_recon_loss, 
                                                                                eval_kl,
                                                                                eval_cls_loss, 
                                                                                eval_cls_acc))
        
        eval_sup_writer.add_scalar('Recon', eval_recon_loss, global_step=step_sup)
        eval_sup_writer.add_scalar('KL Sup', eval_kl, global_step=step_sup)
        eval_sup_writer.add_scalar('CLF Loss', eval_cls_loss, global_step=step_sup)
        eval_sup_writer.add_scalar('CLF Acc', eval_cls_acc, global_step=step_sup)

        # save model every epoch
        torch.save(model.state_dict(), save_path)
        normalizer.save_minmax()

        if step_unsup % 40000 == 0 and step_unsup > 0:
            for p in optimizer.param_groups:
                p['lr'] = args["lr"] / lr_factor
            lr_factor *= 2

        if ep % 1 == 0:
            # plot spectrograms
            audio, onset_pr, frame_pr, emotion_cls = train_ds[10]
            pr_visualize = onset_pr + frame_pr
            pr = torch.cat([onset_pr, frame_pr], dim=-1)
            melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)
            melspec_original = wav_to_melspec(audio)
            
            # use log melspec
            pr = pr.cuda().unsqueeze(0)
            if args["melspec_mode"] == "log":
                melspec = torch.log(melspec + 1e-12).cuda()
            elif args["melspec_mode"] == "log-tanh":
                melspec = torch.nn.Tanh()(0.25 * torch.log(melspec + 1e-12)).cuda()
            elif args["melspec_mode"] == "log-minmax":
                melspec = normalizer.transform(torch.log(melspec + 1e-12)).cuda()
            
            melspec_hat, z, cls_z_logits, cls_z_prob, z_dist = model(melspec, pr)
            
            if args["melspec_mode"] == "log":
                melspec_hat_denorm = torch.exp(melspec_hat).cuda().T.squeeze()
            elif args["melspec_mode"] == "log-tanh":
                def atanh(x):
                    return 0.5*torch.log((1+x)/(1-x) + 1e-12)
                melspec_hat_denorm = torch.exp(atanh(melspec_hat) * 4).cuda()
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
            librosa.display.specshow(melspec_db_2, x_axis='time',
                                    y_axis='mel', sr=16000,
                                    fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            train_unsup_writer.add_figure('spec_recon', fig, global_step=step_unsup, close=True)

            # plot piano rolls
            fig = plt.figure(figsize=(8,8))
            plt.imshow(pr_visualize.squeeze().cpu().detach().numpy().T)
            train_unsup_writer.add_figure('onset_pr_original', fig, global_step=step_unsup, close=True)

            # plot latent space
            z_lst = []
            cls_lst = []
            actual_cls_lst = []
            for i, x_temp in tqdm(enumerate(train_dl), total=len(train_dl), desc='Running latents on train set:'):
                audio, onset_pr, frame_pr, emotion_cls = x_temp
                pr_visualize = onset_pr + frame_pr
                pr = torch.cat([onset_pr, frame_pr], dim=-1)
                melspec = torch.transpose(wav_to_melspec(audio), 1, 2)[:, :-1, :]   # (b, 625, 128)
                melspec_original = wav_to_melspec(audio)
                
                # use log melspec
                pr = pr.cuda().unsqueeze(0)
                if args["melspec_mode"] == "log":
                    melspec = torch.log(melspec + 1e-12).cuda()
                elif args["melspec_mode"] == "log-tanh":
                    melspec = torch.nn.Tanh()(0.25 * torch.log(melspec + 1e-12)).cuda()
                elif args["melspec_mode"] == "log-minmax":
                    melspec = normalizer.transform(torch.log(melspec + 1e-12)).cuda()

                pr = pr.squeeze()
                melspec_hat, z, cls_z_logits, cls_z_prob, z_dist = model(melspec, pr)
                z_lst.append(z.cpu().detach())
                cls_lst.append(torch.argmax(cls_z_logits, dim=-1).squeeze().cpu().detach())
                actual_cls_lst.append(emotion_cls.cpu().detach())
            
            z_lst = torch.cat(z_lst, dim=0).numpy()
            cls_lst = torch.cat(cls_lst, dim=0).cpu().detach().numpy()
            actual_cls_lst = torch.cat(actual_cls_lst, dim=0).cpu().detach().numpy()

            from sklearn.manifold import TSNE
            import seaborn as sns
            sns.set()

            print("Plotting TSNE...", end="\r")
            tsne = TSNE(n_components=2, verbose=0)  #metric='manhattan'
            tsne_features = tsne.fit_transform(z_lst)
            color = cls_lst
            palette = sns.color_palette("bright", len(set(color)))
            fig = plt.figure(figsize=(8,8))
            sns.scatterplot(tsne_features[:,0], tsne_features[:,1], palette=palette, hue=color, legend='full')
            train_unsup_writer.add_figure('tsne_z', fig, global_step=step_unsup, close=True)

            fig = plt.figure(figsize=(8,8))
            color = actual_cls_lst
            palette = sns.color_palette("bright", len(set(color)))
            sns.scatterplot(tsne_features[:,0], tsne_features[:,1], palette=palette, hue=color, legend='full')
            train_unsup_writer.add_figure('tsne_z_actual', fig, global_step=step_unsup, close=True)
            print("Plotting TSNE...done.")



if __name__ == "__main__":

    # housekeeping
    with open('nms_latent_config.json') as f:
        args = json.load(f)
    if not os.path.isdir('params'):
        os.mkdir('params')
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    save_path = 'params/{}_{}.pt'.format(args['name'], datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    save_path += '_{}'.format(args["melspec_mode"])
    save_path += '_emotion'

    NUM_EMOTIONS = 2
    MELSPEC_DIM = 128
    PR_DIM = 88

    # s_percent, u_percent = str(args["percent"]), str(100 - args["percent"])
    # print("Supervised / Unsupervised percentage: {} / {}".format(s_percent, u_percent))

    # load unlabelled data
    train_ds = MAESTRO(path='/data/MAESTRO', groups=['train'], sequence_length=320000)
    train_dl = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, num_workers=0)
    val_ds = MAESTRO(path='/data/MAESTRO', groups=['validation'], sequence_length=320000)
    val_dl = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)
    test_ds = MAESTRO(path='/data/MAESTRO', groups=['test'], sequence_length=320000)
    test_dl = DataLoader(test_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)

    # load labelled data
    # train_s_ds = MAESTRO(path='/data/MAESTRO', groups=['train_s_{}'.format(s_percent)], sequence_length=320000)
    # train_s_dl = DataLoader(train_s_ds, batch_size=args["batch_size"], shuffle=True, num_workers=0)
    # val_s_ds = MAESTRO(path='/data/MAESTRO', groups=['validation_s_{}'.format(s_percent)], sequence_length=320000)
    # val_s_dl = DataLoader(val_s_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)
    # test_s_ds = MAESTRO(path='/data/MAESTRO', groups=['test_s_{}'.format(s_percent)], sequence_length=320000)
    # test_s_dl = DataLoader(test_s_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)

    # load all data
    # train_ds = MAESTRO(path='/data/MAESTRO', groups=['train_all'], sequence_length=320000)
    # train_dl = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True, num_workers=0)
    # val_ds = MAESTRO(path='/data/MAESTRO', groups=['validation_all'], sequence_length=320000)
    # val_dl = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)
    # test_ds = MAESTRO(path='/data/MAESTRO', groups=['test_all'], sequence_length=320000)
    # test_dl = DataLoader(test_ds, batch_size=args["batch_size"], shuffle=False, num_workers=0)

    # load emotion data
    train_emotion_ds = MAESTRO(path='/data/MAESTRO', groups=['train_emotion'], sequence_length=320000)
    train_emotion_dl = DataLoader(train_emotion_ds, batch_size=args["batch_size"] // 4, shuffle=True, num_workers=0)
    val_emotion_ds = MAESTRO(path='/data/MAESTRO', groups=['validation_emotion'], sequence_length=320000)
    val_emotion_dl = DataLoader(val_emotion_ds, batch_size=args["batch_size"] // 4, shuffle=False, num_workers=0)
    test_emotion_ds = MAESTRO(path='/data/MAESTRO', groups=['test_emotion'], sequence_length=320000)
    test_emotion_dl = DataLoader(test_emotion_ds, batch_size=args["batch_size"] // 4, shuffle=False, num_workers=0)

    # load model
    model = NMSLatent(n_component=NUM_EMOTIONS)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), eps=1e-9)

    wav_to_melspec = Spectrogram.MelSpectrogram(sr=16000)
    normalizer = Normalizer(mode="imagewise")

    # load writers
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = 'logs/'+args['name']+'_emotion/'+current_time+'/train'
    eval_log_dir = 'logs/'+args['name']+'_emotion/'+current_time+'/eval'
    train_unsup_writer = SummaryWriter(train_log_dir + "_unsup")
    train_sup_writer = SummaryWriter(train_log_dir + "_sup")
    eval_unsup_writer = SummaryWriter(eval_log_dir + "_unsup")
    eval_sup_writer = SummaryWriter(eval_log_dir + "_sup")
    
    training()