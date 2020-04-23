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
import numpy as np
import time
from sklearn.mixture import GaussianMixture
from torch.distributions import kl_divergence, Normal

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
version = 1
model = GMMMusicTransformerVAE(embedding_dim=args["hidden_dim"], 
                            vocab_size=388+2, 
                            num_layer=args["num_layer"],
                            max_seq=args["max_seq"], 
                            dropout=args["dropout"],
                            filter_size=args["filter_size"],
                            n_component=args["n_component"])
# model = torch.load("params/transformer-gmm-vae-{}.pt".format(version))
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(args["hidden_dim"], optimizer=optimizer, warmup_steps=8000,
                            name="noam")

# optimizer.load_state_dict(torch.load("params/opt-gmm-{}.pt".format(version)))
# scheduler.load_state_dict(torch.load("params/scheduler-gmm-{}.pt".format(version)))


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
train_ds_dist = MaestroDataset(filenames, mode="train")
train_dl_dist = DataLoader(train_ds_dist, batch_size=args["batch_size"], shuffle=is_shuffle, num_workers=0)
val_ds_dist = MaestroDataset(filenames, mode="val")
val_dl_dist = DataLoader(val_ds_dist, batch_size=args["batch_size"], shuffle=False, num_workers=0)
test_ds_dist = MaestroDataset(filenames, mode="test")
test_dl_dist = DataLoader(test_ds_dist, batch_size=args["batch_size"], shuffle=is_shuffle, num_workers=0)
dl = train_dl_dist
print(len(train_ds_dist), len(val_ds_dist), len(test_ds_dist))

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+args['name']+'/'+current_time+'/train'
eval_log_dir = 'logs/'+args['name']+'/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)


def loss_function(out, performance_tokens_padded, dis, qy_x, logLogit_qy_x, step, beta,
                    is_unsupervised=True, y_label=None):
    # anneal beta
    if step < 1000:
        beta0 = 0
    else:
        beta0 = min((step - 1000) / 10000 * beta, beta)

    # reconstruction loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    recon_loss = criterion(out.view(-1, out.shape[-1]), performance_tokens_padded.view(-1))  # autoencoder


    if is_unsupervised:
        # KL latent loss
        kld_lat_total = 0
        n_component = qy_x.shape[-1]

        for k in torch.arange(0, n_component):       # number of components
            # infer current p(z|y)
            mu_pz_y, var_pz_y = model.mu_lookup(k.cuda()), model.logvar_lookup(k.cuda()).exp_()
            dis_pz_y = Normal(mu_pz_y, var_pz_y)
            kld_lat = torch.mean(kl_divergence(dis, dis_pz_y), dim=-1)
            kld_lat *= qy_x[:, k]

            kld_lat_total += kld_lat.mean()
        
        # KL class loss --> KL[q(y|x) || p(y)] = H(q(y|x)) - log p(y)
        def entropy(qy_x, logLogit_qy_x):
            return torch.mean(qy_x * torch.nn.functional.log_softmax(logLogit_qy_x, dim=1), dim=1)
        
        h_qy_x = entropy(qy_x, logLogit_qy_x)
        kld_cls = (h_qy_x - np.log(1 / n_component)).mean()

        loss = recon_loss + beta0 * (kld_lat_total + kld_cls)

        return loss, recon_loss, kld_lat_total, kld_cls, h_qy_x.mean()

    else:
        mu_pz_y, var_pz_y = model.mu_lookup(y_label.cuda().long()), model.logvar_lookup(y_label.cuda().long()).exp_()
        dis_pz_y = Normal(mu_pz_y, var_pz_y)
        kld_lat_total = torch.mean(kl_divergence(dis, dis_pz_y), dim=-1).mean()

        label_clf_loss = nn.CrossEntropyLoss()(qy_x, y_label.cuda().long())
        loss = recon_loss + beta0 * (kld_lat_total) + label_clf_loss

        return loss, recon_loss, kld_lat_total, torch.Tensor([0]), torch.Tensor([0])


def training_phase(model, optimizer, scheduler):

    step = 0

    # plot the very initial tsne first
    z_lst = []
    cluster_lst = []
    for j, x in tqdm(enumerate(train_dl_dist), total=len(train_dl_dist)):
        performance_tokens = x
        performance_tokens = performance_tokens.cuda().long()
        optimizer.zero_grad()
        out, dis, z, logLogit_qy_x, qy_x, y = model(performance_tokens)

        z_lst.append(z.detach())
        cluster_lst.append(y.detach())

    z_lst = torch.cat(z_lst, dim=0).cpu().detach()
    cluster_lst = torch.cat(cluster_lst, dim=0).cpu().numpy()
    mu_lst = []
    for i in range(4):
        mu_lst.append(model.mu_lookup(torch.Tensor([i]).long().cuda()))

    mu_lst = torch.cat(mu_lst, dim=0).cpu()
    z_lst = torch.cat([z_lst, mu_lst]).cpu().detach().numpy()
    
    tsne = TSNE(n_components=2, verbose=3)  #metric='manhattan'
    tsne_features = tsne.fit_transform(z_lst.squeeze())

    color = cluster_lst
    palette = sns.color_palette("bright", len(set(color)))
    fig = plt.figure(figsize=(8,8))
    sns.scatterplot(tsne_features[:954,0], tsne_features[:954,1], palette=palette, hue=color, legend='full')
    sns.scatterplot(tsne_features[954:,0], tsne_features[954:,1], marker="X", color="black")
    train_summary_writer.add_figure('tsne', fig, global_step=step, close=True)

    
    for i in range(1, args['n_epochs'] + 1):
        print("Epoch {} / {}".format(i, args['n_epochs']))

        batch_loss = 0
        batch_acc = 0
        test_loss = 0
        test_kl = 0
        test_acc = 0

        len_dl = len(train_dl_dist)

        # get pseudo label by using any clustering algorithm
        print("Getting pseudo label...")
        z_lst = []
        for j, x in tqdm(enumerate(train_dl_dist), total=len(train_dl_dist)):
            performance_tokens = x
            performance_tokens = performance_tokens.cuda().long()
            out, dis, z, logLogit_qy_x, qy_x, y = model(performance_tokens)
            z_lst.append(z.cpu().detach())
        z_lst = torch.cat(z_lst, dim=0).cpu().detach().numpy()

        # use GMM for clustering
        gmm = GaussianMixture(n_components=4)
        gmm.fit(z_lst)
        clusters = gmm.predict(z_lst).squeeze()
        print(clusters)

        z_lst = []
        cluster_lst = []

        for j, x in enumerate(train_dl_dist):
            print(j, "/", len_dl, end="\r")
            performance_tokens = x
            performance_tokens = performance_tokens.cuda().long()
            # performance_tokens = performance_tokens.cuda(model.output_device).long()
            # melody_tokens = melody_tokens.cuda().long()

            optimizer.zero_grad()
            out, dis, z, logLogit_qy_x, qy_x, y = model(performance_tokens)

            z_lst.append(z.detach())
            cluster_lst.append(y.detach())

            performance_tokens_padded = F.pad(input=performance_tokens, 
                                              pad=(0, 1, 0, 0), mode='constant', value=1) 

            # loss function
            p = 0
            if random.random() <= p:
                # train unsupervised
                loss, recon_loss, kld_lat_total, kld_cls, h_qy_x = loss_function(out, performance_tokens_padded, 
                                                                        dis, qy_x, logLogit_qy_x, step, args['beta'])
            else:
                # train pseudo supervised
                y_label = gmm.predict(z.cpu().detach().numpy()).squeeze()
                y_label = torch.Tensor(y_label).cuda().long()
                loss, recon_loss, kld_lat_total, kld_cls, h_qy_x = loss_function(out, performance_tokens_padded, 
                                                                        dis, qy_x, logLogit_qy_x, step, args['beta'],
                                                                        is_unsupervised=False, y_label=y_label)
                
            acc = accuracy_score(performance_tokens_padded.view(-1).cpu().detach().numpy(),
                                torch.argmax(out, dim=-1).view(-1).cpu().detach().numpy())

            temp = qy_x.cpu().detach().numpy()
            qy_x_str = "[{:.2f} {:.2f} {:.2f} {:.2f}]".format(temp[0][0],
                                                                temp[0][1],
                                                                temp[0][2],
                                                                temp[0][3])
            qy_x_str_2 = "[{:.2f} {:.2f} {:.2f} {:.2f}]".format(temp[1][0],
                                                                temp[1][1],
                                                                temp[1][2],
                                                                temp[1][3])

            print("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {}".format(step, loss.item(), recon_loss.item(), 
                                                                    kld_lat_total.item(), kld_cls.item(), 
                                                                    acc, qy_x_str, qy_x_str_2), 
                                                                    end="\r")
            
            loss.backward()
            scheduler.step()
            batch_loss += loss.item()
            batch_acc += acc
        
            train_summary_writer.add_scalar('loss', loss.item(), global_step=step)
            train_summary_writer.add_scalar('kl_loss', kld_lat_total.item() + kld_cls.item(), global_step=step)
            train_summary_writer.add_scalar('acc', acc, global_step=step)
            train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=step)

            if step % 250 == 0:

                print("Saving model", "params/transformer-vae-gmm-{}.pt".format(version))
                # for model, save the whole thing
                torch.save(single_model, "params/transformer-gmm-vae-{}.pt".format(version))
                torch.save(optimizer.state_dict(), "params/opt-gmm-{}.pt".format(version))
                torch.save(scheduler.state_dict(), "params/scheduler-gmm-{}.pt".format(version))
                
                print("Evaluation...")
                single_model.eval()

                # evaluate on vgmidi
                for j, x in tqdm(enumerate(val_dl_dist), total=len(val_dl_dist)):
                    
                    performance_tokens = x
                    performance_tokens = performance_tokens.cuda().long()
                    # performance_tokens = performance_tokens.cuda(model.output_device).long()

                    out, dis, z, logLogit_qy_x, qy_x, y = single_model(performance_tokens)

                    performance_tokens_padded = F.pad(input=performance_tokens, 
                                                    pad=(0, 1, 0, 0), mode='constant', value=1)
                    
                    loss, recon_loss, kld_lat_total, kld_cls, h_qy_x = loss_function(out, performance_tokens_padded, 
                                                                    dis, qy_x, logLogit_qy_x, step, args['beta'])

                    acc = accuracy_score(performance_tokens_padded.view(-1).cpu().detach().numpy(),
                                        torch.argmax(out, dim=-1).view(-1).cpu().detach().numpy())

                    test_loss += recon_loss.item()
                    test_kl += kld_lat_total.item() + kld_cls.item()
                    test_acc += acc
                
                eval_summary_writer.add_scalar('loss', test_loss / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('kl_loss', kld_lat_total.item() + kld_cls.item(), global_step=step)
                eval_summary_writer.add_scalar('acc', test_acc / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=step)

                print("Evaluation: {:.5f}  {:.5f}  {:.5f}".format(test_loss / len(val_dl_dist), 
                                                        test_kl / len(val_dl_dist),
                                                        test_acc / len(val_dl_dist)))
                test_loss = 0
                test_acc = 0
            
            step += 1
            model.train()
        

        # plot tsne after each epoch
        print("Plotting TSNE...")
        z_lst = torch.cat(z_lst, dim=0).cpu().detach()
        cluster_lst = torch.cat(cluster_lst, dim=0).cpu().numpy()
        mu_lst = []
        for i in range(4):
            mu_lst.append(model.mu_lookup(torch.Tensor([i]).long().cuda()))

        mu_lst = torch.cat(mu_lst, dim=0).cpu()
        z_lst = torch.cat([z_lst, mu_lst]).cpu().detach().numpy()
        
        tsne = TSNE(n_components=2, verbose=3)  #metric='manhattan'
        tsne_features = tsne.fit_transform(z_lst.squeeze())

        color = cluster_lst
        palette = sns.color_palette("bright", len(set(color)))
        fig = plt.figure(figsize=(8,8))
        sns.scatterplot(tsne_features[:954,0], tsne_features[:954,1], palette=palette, hue=color, legend='full')
        sns.scatterplot(tsne_features[954:,0], tsne_features[954:,1], marker="X", color="black")
        train_summary_writer.add_figure('tsne', fig, global_step=step, close=True)
        

def evaluation_phase(model, optimizer, scheduler):
    model.eval()

    def run(dl):
        test_loss = 0
        test_recon_loss = 0
        test_kld_loss = 0
        test_acc = 0

        for j, x in enumerate(dl):

            performance_tokens = x
            performance_tokens = performance_tokens.cuda().long()
            # performance_tokens = performance_tokens.cuda(model.output_device).long()

            out, dis, z, logLogit_qy_x, qy_x, y = single_model(performance_tokens)

            performance_tokens_padded = F.pad(input=performance_tokens, 
                                            pad=(0, 1, 0, 0), mode='constant', value=1)
            
            loss, recon_loss, kld_lat_total, kld_cls = loss_function(out, performance_tokens_padded, 
                                                            dis, qy_x, logLogit_qy_x, step, args['beta'])

            acc = accuracy_score(performance_tokens_padded.view(-1).cpu().detach().numpy(),
                                torch.argmax(out, dim=-1).view(-1).cpu().detach().numpy())

            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kld_loss += kld_lat_total.item() + kld_cls.item()
            test_acc += acc
        
        print('evaluate loss: {:.5f}'.format(test_loss / len(dl)))
        print('evaluate recon: {:.5f}'.format(test_recon_loss / len(dl)))
        print('evaluate kld: {:.5f}'.format(test_kld_loss / len(dl)))
        print('evaluate acc: {:.5f}'.format(test_acc / len(dl)))
    
    run(test_dl_dist)
    # run(train_dl_dist)
    run(val_dl_dist)



training_phase(model, optimizer, scheduler)
evaluation_phase(model, optimizer, scheduler)
# testing(model, optimizer, scheduler)