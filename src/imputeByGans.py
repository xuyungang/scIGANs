## last update: 2019/04/28

from __future__ import print_function, division
import argparse
import os
import numpy as np
import pandas as pd
import math
import sys
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

if  os.path.isdir('GANs_models')!=True:
    os.makedirs('GANs_models')

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--kt', type=float, default=0, help='kt parameters')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameters')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
#parser.add_argument('--n_cpu', type=int, default=20, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=100, help='size of each image dimension')
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
parser.add_argument('--threthold', type=float, default=0.01, help='the convergence threthold')

max_ncls=16  # 

opt = parser.parse_args()
#opt.impute=True
#print(opt)
model_basename = os.path.basename(opt.file_d)+"-"+os.path.basename(opt.file_c)+"-"+str(opt.latent_dim)+"-"+str(opt.n_epochs)+"-"+str(opt.ncls)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
#%% for debug use only
#opt.file_d='ercc.csv'
#opt.file_c='ercc.label.txt'
#opt.img_size=9
#opt.train=True
#opt.n_epochs = 1
#cuda = False
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
        d = pd.read_csv(cls_file,header=None,index_col=False)  #
        self.data_cls = pd.Categorical(d.iloc[:,0]).codes      #
        self.transform = transform
        self.fig_h = opt.img_size ##
        

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
    # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data.iloc[:,idx].values[0:(self.fig_h*self.fig_h),].reshape(self.fig_h,self.fig_h,1).astype('double')  #
        label = np.array(self.data_cls[idx]).astype('int32')         #
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
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.cn1=32
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1*(self.init_size**2)))
        self.l1p = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1*(opt.img_size**2)))


        
        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
#            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),            
        )
        

        
        self.conv_blocks_02p = nn.Sequential(
#            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=opt.img_size),#torch.Size([bs, 128, 16, 16])
            nn.Conv2d(max_ncls,  self.cn1//4, 3, stride=1, padding=1),#torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d( self.cn1//4),
            nn.ReLU(),           
        )

                
        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(40, 0.8),
            nn.Conv2d(40, self.cn1, 3, stride=1, padding=1),#torch.Size([bs, 1, 32, 32])
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, opt.channels, 3, stride=1, padding=1),#torch.Size([bs, 1, 32, 32])
            nn.Sigmoid()
        )
    def forward(self, noise,label_oh):
        out = self.l1p(noise)        
        out = out.view(out.shape[0], self.cn1, opt.img_size, opt.img_size)
        out01 = self.conv_blocks_01p(out) #([4, 32, 124, 124])
#        
        label_oh=label_oh.unsqueeze(2)
        label_oh=label_oh.unsqueeze(2)  
        out02 = self.conv_blocks_02p(label_oh) #([4, 8, 124, 124])
        
       
        out1=torch.cat((out01,out02),1)
        out1=self.conv_blocks_1(out1)
        return out1
        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.cn1=32
        self.down_size0 = 64
        self.down_size = 32
        #pre
        self.pre= nn.Sequential(
            nn.Linear(opt.img_size**2,self.down_size0**2),
        )

        # Upsampling
        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, self.cn1, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.cn1),            
            nn.ReLU(),
            nn.Conv2d(self.cn1, self.cn1//2, 3, 1, 1),
#            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.cn1//2),            
            nn.ReLU(),
        )
        
        
        self.conv_blocks02p = nn.Sequential(
#            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=self.down_size),#torch.Size([bs, 128, 16, 16])
            nn.Conv2d(max_ncls,  self.cn1//4, 3, stride=1, padding=1),#torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d( self.cn1//4),
            nn.ReLU(),
        )
        
        # Fully-connected layers
        
        down_dim =24 * (self.down_size)**2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU()
        )
        # Upsampling 32X32
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, opt.channels, 2, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img,label_oh):
        
        out00 = self.pre(img.view((img.size()[0],-1))).view((img.size()[0],1,self.down_size0,self.down_size0))
        out01 = self.down(out00)#([4, 16, 32, 32])
        
        label_oh=label_oh.unsqueeze(2)
        label_oh=label_oh.unsqueeze(2)
        out02 = self.conv_blocks02p(label_oh)#([4, 16, 32, 32])
####        
        out1=torch.cat((out01,out02),1)        
######        
        out = self.fc(out1.view(out1.size(0), -1))
        out = self.up(out.view(out.size(0), 24, self.down_size, self.down_size))
        return out

       
#%%
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
    print("scIGANs is runing on GPUs.")
else:
    print("scIGANs is runing on CPUs.")
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
    model_exists = os.path.isfile('GANs_models/'+model_basename+'-g.pt')
    if model_exists:
        overwrite = input("WARNING: A trained model exists with the same settings for your data.\n         Do you want to train and overwrite it?: (y/n)\n")
        if overwrite != "y": 
            print("The training was deprecated since optical model exists.")
            print("scIGANs continues imputation using existing model...")
            sys.exit()# if model exists and do not want to train again, exit the program
    print("The optimal model will be output in \""+os.getcwd()+"/GANs_models\" with basename = " + model_basename)
#    if opt.dpt!='' and cuda==True:
#        discriminator.load_state_dict(torch.load(opt.dpt))    
#        generator.load_state_dict(torch.load(opt.gpt))
#    if opt.dpt!='' and cuda != True:
#        discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))    
#        generator.load_state_dict(torch.load(opt.gpt, map_location=lambda storage, loc: storage))
    max_M = sys.float_info.max
    min_dM = 0.001
    dM =  1
    for epoch in range(opt.n_epochs):
        cur_M = 0
        cur_dM = 1
        for i, batch_sample in enumerate(dataloader):
#            if i==0:
#                break
            imgs = batch_sample['data'].type(Tensor)
            label= batch_sample['label']
            label_oh = one_hot((label).type(torch.LongTensor),max_ncls).type(Tensor)  #
    
    
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
            cur_M += M
            #--------------
            # Log Progress
            #--------------
    
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, delta_M: %f,k: %f" % (epoch+1, opt.n_epochs, i+1, len(dataloader),
                                                                np.asscalar(d_loss.detach().data.cpu().numpy()), np.asscalar(g_loss.detach().data.cpu().numpy()),
                                                                M, dM, k))
    
            batches_done = epoch * len(dataloader) + i
        #get the M of current epoch
        cur_M = cur_M/len(dataloader)
        if cur_M < max_M: #if current model is better than previous one
            torch.save(discriminator.state_dict(),'GANs_models/'+model_basename+'-d.pt')
            torch.save(generator.state_dict(),'GANs_models/'+model_basename+'-g.pt')
            dM = min(max_M-cur_M,cur_M)
            if dM < min_dM: # if convergence threthold meets, stop training
                print("Training was stopped after " + str(epoch+1)+" epoches since the convergence threthold ("+str(min_dM)+".) reached: " + str(dM))
                break
            cur_dM = max_M-cur_M
            max_M =  cur_M
        if epoch+1 == opt.n_epochs and cur_dM > min_dM:
            print("Training was stopped after " + str(epoch+1)+" epoches since the maximum epoches reached: "+str(opt.n_epochs)+".")
            print("WARNING: the convergence threthold ("+str(min_dM)+") was not met. Current value is: "+str(cur_dM))
            print("You may need more epoches to get the most optimal model!!!")

if opt.impute:
    if opt.gpt=='':
        model_g = 'GANs_models/'+model_basename+'-g.pt'
        model_exists = os.path.isfile(model_g)
        if not model_exists: 
            print("ERROR: There is no model exists with the given settings for your data.")
            print("Please set --train instead of --impute to train a model fisrt.")
            sys.exit("scIGANs stopped!!!")# if model exists and do not want to train again, exit the program
            print()
    else:
        model_g = opt.gpt
    print(model_g+" is used for imputation.")
    if cuda==True:
        #discriminator.load_state_dict(torch.load(opt.dpt))    
        generator.load_state_dict(torch.load(model_g))
    else:
        #discriminator.load_state_dict(torch.load(opt.dpt, map_location=lambda storage, loc: storage))    
        generator.load_state_dict(torch.load(model_g, map_location=lambda storage, loc: storage))
######################################################
###        impute by type
######################################################
    sim_size=opt.sim_size
    sim_out=list()
    for i in range(opt.ncls):
        label_oh = one_hot(torch.from_numpy(np.repeat(i,sim_size)).type(torch.LongTensor),max_ncls).type(Tensor)
       
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (sim_size, opt.latent_dim))))
    
        # Generate a batch of images
        fake_imgs = generator(z,label_oh).detach().data.cpu().numpy()
        sim_out.append(fake_imgs)    
    mydataset = MyDataset(d_file=opt.file_d,
                                cls_file=opt.file_c)    
    data_imp_org=np.asarray([mydataset[i]['data'].reshape((opt.img_size*opt.img_size)) for i in range(len(mydataset))]).T
    data_imp=data_imp_org.copy()
    
    #by type
    sim_out_org=sim_out
    rels = [my_knn_type(data_imp_org[:,k],sim_out_org[int(mydataset[k]['label'])-1],knn_k=opt.knn_k) for k in range(len(mydataset))]         
    pd.DataFrame(rels).to_csv(os.path.dirname(os.path.abspath(opt.file_d))+'/scIGANs-'+os.path.basename(os.path.abspath(opt.file_d))+'.csv') #imped data  

