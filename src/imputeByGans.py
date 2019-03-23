from __future__ import print_function, division
import argparse
import os
import numpy as np
import pandas as pd
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime

if  os.path.isdir('images')!=True:
    os.makedirs('images')

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--kt', type=float, default=0, help='kt parameters')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameters')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=20, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=124, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default= 1, help='number of training steps for discriminator per iter')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--dpt', type=str, default='', help='load discrimnator model')
parser.add_argument('--gpt', type=str, default='', help='load generator model')
parser.add_argument('--train', help='train the network', action='store_true')
parser.add_argument('--impute', help='do imputation', action='store_true')
parser.add_argument('--sim_size', type=int, default=200, help='number of sim_imgs in each type')
parser.add_argument('--file_d', type=str, default='', help='path of data file')
parser.add_argument('--file_c', type=str, default='', help='path of cls file')
parser.add_argument('--ncls', type=int, default=4, help='number of clusters')
parser.add_argument('--knn_k', type=int, default=10, help='neighours used')
parser.add_argument('--lr_rate', type=int, default=10, help='rate for slow learning')


opt = parser.parse_args()
#opt.impute=True
print(opt)
prestr=datetime.now().strftime('-%m%d%H%M-')
print(prestr)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

#%%
class MyDataset(Dataset):
    """Operations with the datasets."""

    def __init__(self, d_file, cls_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(d_file,header=0,index_col=0)
        self.data_cls = pd.read_csv(cls_file,header=0,index_col=0)
        self.transform = transform
        self.fig_h = int(math.sqrt(self.data.shape[0]))
        

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
    # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data.iloc[:,idx].as_matrix().reshape(self.fig_h,self.fig_h,1).astype('double')
        label = self.data_cls.iloc[idx, :].as_matrix().astype('int')
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data,label = sample['data'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((2, 0, 1))

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label)                
                }

def one_hot(batch,depth):
    ones = torch.eye(depth)
    return ones.index_select(0,batch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
#%%
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.cn1=32
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1*self.init_size**2))
        

        self.conv_blocks_01 = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.cn1, 2*self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(2*self.cn1, 0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_blocks_02 = nn.Sequential(
#            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=16),#torch.Size([bs, 128, 16, 16])
            nn.Conv2d(opt.ncls,  self.cn1, 3, stride=1, padding=1),#torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d( self.cn1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),#torch.Size([bs, 128, 32, 32])
            nn.Conv2d( self.cn1, self.cn1//2, 3, stride=1, padding=0),#torch.Size([bs, 64, 32, 32])
            nn.BatchNorm2d( self.cn1//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),#torch.Size([bs, 128, 32, 32])
            nn.Conv2d( self.cn1//2,  self.cn1//4, 3, stride=1, padding=1),#torch.Size([bs, 64, 32, 32])
            nn.BatchNorm2d( self.cn1//4),
            nn.ReLU(),

            

#            nn.Tanh()
        )
        
        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(40, 0.8),
            nn.Conv2d(40, self.cn1, 3, stride=1, padding=1),#torch.Size([bs, 1, 32, 32])
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, opt.channels, 3, stride=1, padding=1),#torch.Size([bs, 1, 32, 32])
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Tanh()
            nn.Sigmoid()
        )
    def forward(self, noise,label_oh):
        out = self.l1(noise)        
        out = out.view(out.shape[0], self.cn1, self.init_size, self.init_size)
        out01 = self.conv_blocks_01(out) #([4, 32, 124, 124])
        
        label_oh=label_oh.unsqueeze(2)
        label_oh=label_oh.unsqueeze(2)
        out02 = self.conv_blocks_02(label_oh) #([4, 8, 124, 124])
##        
        out1=torch.cat((out01,out02),1)
        out1=self.conv_blocks_1(out1)
        return out1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.cn1=32
        #pre
        self.pre= nn.Sequential(
            nn.Linear(opt.img_size**2,opt.img_size**2),
        )

        # Upsampling
        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, self.cn1, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.cn1),            
            nn.ReLU(),
            nn.Conv2d(self.cn1, self.cn1//2, 3, 1, 2),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.cn1//2),            
            nn.ReLU(),
        )
        self.conv_blocks02 = nn.Sequential(
#            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=8),#torch.Size([bs, 128, 16, 16])
            nn.Conv2d(opt.ncls, self.cn1, 3, stride=1, padding=1),#torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),#torch.Size([bs, 128, 32, 32])
            nn.Conv2d(self.cn1, self.cn1//2, 3, stride=1, padding=1),#torch.Size([bs, 64, 32, 32])
        )
        # Fully-connected layers
        self.down_size = 32
        down_dim = 32 * (self.down_size)**2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU()
        )
        # Upsampling
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(32, 16, 3, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, opt.channels, 3, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img,label_oh):
        
        out00 = self.pre(img.view((img.size()[0],-1))).view((img.size()[0],1,opt.img_size,opt.img_size))
#        out00 = img
        out01 = self.down(out00)#([4, 16, 32, 32])
        
        label_oh=label_oh.unsqueeze(2)
        label_oh=label_oh.unsqueeze(2)
        out02 = self.conv_blocks02(label_oh)#([4, 16, 32, 32])
###        
        out1=torch.cat((out01,out02),1)        
####        
        out = self.fc(out1.view(out1.size(0), -1))
        out = self.up(out.view(out.size(0), 32, self.down_size, self.down_size))
        return out


def my_knn_type(data_imp_org_k,sim_out_k,knn_k=10):
        sim_size=sim_out_k.shape[0]
        out=data_imp_org_k.copy()
        q1k = data_imp_org_k.reshape((opt.img_size*opt.img_size,1))       
        q1kl = np.int8(q1k>0) # get which part in cell k is >0
        q1kn = np.repeat(q1k*q1kl,repeats=sim_size,axis=1) # get >0 part of cell k  
        sim_out_tmp=sim_out_k.reshape((sim_size,opt.img_size*opt.img_size)).T
        sim_outn = sim_out_tmp * np.repeat(q1kl,repeats=sim_size,axis=1)  # get the >0 part of simmed ones            
        diff = q1kn-sim_outn #distance of cell k to simmed ones
        diff = diff*diff  
        rel = np.sum(diff,axis=0)
        locs = np.where(q1kl==0)[0]
#        locs1 = np.where(q1kl==1)[0]
        sim_out_c=np.median(sim_out_tmp[:,rel.argsort()[0:knn_k]],axis=1)    
        out[locs]=sim_out_c[locs]
        return out
#%%
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


# Configure data loader
transformed_dataset = MyDataset(d_file=opt.file_d,
                                cls_file=opt.file_c,
                                           transform=transforms.Compose([
#                                               Rescale(256),
#                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size,
                        shuffle=True, num_workers=0,drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#%%
# ----------
#  Training
# ----------

# BEGAN hyper parameters
gamma = opt.gamma
lambda_k = 0.001
k = opt.kt


if opt.train:    
    if opt.dpt!='' and cuda==True:
        discriminator.load_state_dict(torch.load(opt.dpt))    
        generator.load_state_dict(torch.load(opt.gpt))
    if opt.dpt!='' and cuda != True:
        discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))    
        generator.load_state_dict(torch.load(opt.gpt, map_location=lambda storage, loc: storage))
    
    for epoch in range(opt.n_epochs):
        for i, batch_sample in enumerate(dataloader):
#            if i==0:
#                break
            imgs = batch_sample['data'].type(Tensor)
            label= batch_sample['label']
            label_oh = one_hot((label[:,0]-1).type(torch.LongTensor),opt.ncls).type(Tensor)
    
    
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
    
            # -----------------
            #  Train Generator
            # -----------------
    
            optimizer_G.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
            # Generate a batch of images
            gen_imgs = generator(z,label_oh)
    
            # Loss measures generator's ability to fool the discriminator
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs,label_oh) - gen_imgs))
    
            g_loss.backward()
            optimizer_G.step()
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            d_real = discriminator(real_imgs,label_oh)
            d_fake = discriminator(gen_imgs.detach(),label_oh)
    
            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake
    
            d_loss.backward()
            optimizer_D.step()
    
            #----------------
            # Update weights
            #----------------
    
            diff = torch.mean(gamma * d_loss_real - d_loss_fake)
    
            # Update weight term for fake samples
            k = k + lambda_k *  np.asscalar(diff.detach().data.cpu().numpy())
            k = min(max(k, 0), 1) # Constraint to interval [0, 1]
    
            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).item()
    
            #--------------
            # Log Progress
            #--------------
    
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                np.asscalar(d_loss.detach().data.cpu().numpy()), np.asscalar(g_loss.detach().data.cpu().numpy()),
                                                                M, k))
    
            batches_done = epoch * len(dataloader) + i
        print(prestr+str(epoch)+'.pt')
        torch.save(discriminator.state_dict(),'images/d'+prestr+str(epoch)+'.pt')
        torch.save(generator.state_dict(),'images/g'+prestr+str(epoch)+'.pt')
            

if opt.impute:

    
    if opt.dpt!='' and cuda==True:
        discriminator.load_state_dict(torch.load(opt.dpt))    
        generator.load_state_dict(torch.load(opt.gpt))
    if opt.dpt!='' and cuda != True:
        discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))    
        generator.load_state_dict(torch.load(opt.gpt, map_location=lambda storage, loc: storage))
######################################################
###        imp by type
######################################################
    sim_size=opt.sim_size
    sim_out=list()
    for i in range(opt.ncls):
        label_oh = one_hot(torch.from_numpy(np.repeat(i,sim_size)).type(torch.LongTensor),opt.ncls).type(Tensor)
       
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (sim_size, opt.latent_dim))))
    
        # Generate a batch of images
        fake_imgs = generator(z,label_oh).detach().data.cpu().numpy()
        sim_out.append(fake_imgs)    
    print('imputing...')
    mydataset = MyDataset(d_file=opt.file_d,
                                cls_file=opt.file_c)    
    data_imp_org=np.asarray([mydataset[i]['data'].reshape((opt.img_size*opt.img_size)) for i in range(len(mydataset))]).T
    data_imp=data_imp_org.copy()
    
    #by type
    sim_out_org=sim_out
    rels = [my_knn_type(data_imp_org[:,k],sim_out_org[int(mydataset[k]['label'])-1],knn_k=opt.knn_k) for k in range(len(mydataset))]         
    pd.DataFrame(rels).to_csv(os.path.dirname(opt.file_d)+'/scIGANs-'+os.path.basename(opt.file_d)+'.csv') #imped data  
