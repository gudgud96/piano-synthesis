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
version = 6
model = MusicTransformerVAE(embedding_dim=args["hidden_dim"], 
                            vocab_size=388+2, 
                            num_layer=args["num_layer"],
                            max_seq=args["max_seq"], 
                            dropout=args["dropout"],
                            filter_size=args["filter_size"])
# model = torch.load("params-10042020//transformer-vae-{}.pt".format(version))
# model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(args["hidden_dim"], optimizer=optimizer, warmup_steps=8000,
                            name="noam")

# optimizer.load_state_dict(torch.load("params-10042020//opt-{}.pt".format(version)))
# scheduler.load_state_dict(torch.load("params-10042020//scheduler-{}.pt".format(version)))


if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

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

def training_phase(model, optimizer, scheduler):

    # model = MusicTransformerVAE(embedding_dim=args["hidden_dim"], vocab_size=388+2, num_layer=6,
    #                     max_seq=2048, dropout=0.2)
    # model.cuda()
    # model.train()

    step = 0
    
    for i in range(1, args['n_epochs'] + 1):
        print("Epoch {} / {}".format(i, args['n_epochs']))

        batch_loss = 0
        batch_acc = 0
        test_loss = 0
        test_acc = 0

        len_dl = len(train_dl_dist)

        for j, x in enumerate(train_dl_dist):

            print(j, "/", len_dl, end="\r")
            performance_tokens = x
            performance_tokens = performance_tokens.cuda().long()
            # performance_tokens = performance_tokens.cuda(model.output_device).long()
            # melody_tokens = melody_tokens.cuda().long()

            optimizer.zero_grad()
            out = model(performance_tokens)

            performance_tokens_padded = F.pad(input=performance_tokens, 
                                              pad=(0, 1, 0, 0), mode='constant', value=1) 

            criterion = nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(out.view(-1, out.shape[-1]), performance_tokens_padded.view(-1))  # autoencoder
            acc = accuracy_score(performance_tokens_padded.view(-1).cpu().detach().numpy(),
                                torch.argmax(out, dim=-1).view(-1).cpu().detach().numpy())

            print(step, loss.item(), acc, end="\r")
            
            loss.backward()
            scheduler.step()
            batch_loss += loss.item()
            batch_acc += acc
        
            train_summary_writer.add_scalar('loss', loss.item(), global_step=step)
            train_summary_writer.add_scalar('acc', acc, global_step=step)
            train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=step)

            if step % 250 == 0:

                print("Saving model", "params/transformer-vae-{}.pt".format(version))
                # for model, save the whole thing
                torch.save(single_model, "params/transformer-vae-{}.pt".format(version))
                torch.save(optimizer.state_dict(), "params/opt-{}.pt".format(version))
                torch.save(scheduler.state_dict(), "params/scheduler-{}.pt".format(version))
                
                print("Evaluation...")
                single_model.eval()

                # evaluate on vgmidi
                for j, x in tqdm(enumerate(val_dl_dist), total=len(val_dl_dist)):
                    
                    performance_tokens = x
                    performance_tokens = performance_tokens.cuda().long()
                    # performance_tokens = performance_tokens.cuda(model.output_device).long()

                    out = single_model(performance_tokens)

                    performance_tokens_padded = F.pad(input=performance_tokens, 
                                                    pad=(0, 1, 0, 0), mode='constant', value=1)

                    criterion = nn.CrossEntropyLoss(reduction='mean')
                    loss = criterion(out.view(-1, out.shape[-1]), performance_tokens_padded.view(-1))  # autoencoder

                    acc = accuracy_score(performance_tokens_padded.view(-1).cpu().detach().numpy(),
                                        torch.argmax(out, dim=-1).view(-1).cpu().detach().numpy())

                    test_loss += loss.item()
                    test_acc += acc
                
                eval_summary_writer.add_scalar('loss', test_loss / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('acc', test_acc / len(val_dl_dist), global_step=step)
                eval_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=step)

                print("Evaluation: {:.5f}  {:.5f}".format(test_loss / len(val_dl_dist), 
                                                        test_acc / len(val_dl_dist)))
                test_loss = 0
                test_acc = 0
            
            step += 1
            model.train()
        
        # switch output device to: gpu-1 ~ gpu-n
        # sw_start = time.time()
        # model.output_device = i % (torch.cuda.device_count() -1) + 1
        # sw_end = time.time()
        # print('output switch time: {}'.format(sw_end - sw_start) )
        
        # print('batch loss: {:.5f}  {:.5f}'.format(batch_loss / len(train_dl_dist),
        #                                           test_loss / len(val_dl_dist)))
        # print('batch acc: {:.5f}  {:.5f}'.format(batch_acc / len(train_dl_dist),
        #                                           test_acc / len(val_dl_dist)))

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
            
            test_loss += loss.item()
            test_acc += acc
        
        print('evaluate loss: {:.5f}'.format(test_loss / len(dl)))
        print('evaluate acc: {:.5f}'.format(test_acc / len(dl)))
    
    run(test_dl_dist)
    # run(train_dl_dist)
    run(val_dl_dist)


def testing(model, optimizer, scheduler):
    a = [i % 345 for i in range(2048)]
    a = torch.Tensor(a).cuda().unsqueeze(0).long()
    out = model(a)
    print(torch.argmax(out, dim=-1))



training_phase(model, optimizer, scheduler)
evaluation_phase(model, optimizer, scheduler)
# testing(model, optimizer, scheduler)