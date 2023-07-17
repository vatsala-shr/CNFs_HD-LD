import nibabel as nib
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
import copy
import random
import pickle
random.seed(0)

new_range=0
def set_new_range(num):
    global new_range
    new_range=num

def read_files(file_name, noise_level, data_path):
    ld = nib.load(os.path.join(data_path,'data_ld',file_name+'_'+str(noise_level)+'.nii.gz')).get_fdata().astype('float32')
    if 'blur' in file_name:
        file_name = file_name[file_name.find('_')+1:]
    hd = nib.load(os.path.join(data_path,'data_hd',file_name+'.nii.gz')).get_fdata().astype('float32')
    return ld, hd

def load_data(args):
    global new_range
    new_range=args.new_range
    data_path = '../../Pet_CT/data_masked/'
        
    if args.new_range==1:
        data_path = data_path+'new_range/'
    elif args.new_range==2:
        data_path = data_path+'new_range2/'
      
    if args.blur_mode=='petct':
        file_name = ['blur_pet','blur_ct']
    elif args.blur_mode=='pet':
        file_name = ['blur_pet','ct']
    elif args.blur_mode is None:
        file_name = ['pet','ct']
    else:
        print("INVALID blur mode")

    if args.mode=='pet':
        ld, hd = read_files(file_name[0], args.noise_level[0], data_path)
        ld = np.expand_dims(ld, 1)
        hd = np.expand_dims(hd, 1)
    elif args.mode =='ct':
        ld, hd = read_files(file_name[1], args.noise_level[0], data_path)
        ld = np.expand_dims(ld, 1)
        hd = np.expand_dims(hd, 1)
    elif args.mode == 'petct':  
        ld_pt, hd_pt = read_files(file_name[0], args.noise_level[0], data_path)
        ld_ct, hd_ct = read_files(file_name[1], args.noise_level[1], data_path)
        hd = np.stack((hd_pt,hd_ct),axis=1)
        ld = np.stack((ld_pt,ld_ct),axis=1)
    
    train_size = args.train_size if args.semi_sup else int(args.supervision*args.train_size) #smaller train size when fully supervised 
    
    hd_train = hd[:train_size]
    ld_train = ld[:train_size]
    hd_test = hd[500:]
    ld_test = ld[500:]
        
    if args.semi_sup and args.supervision<1.0 and args.secondary_noisy>0:
        noisy_target = get_noisy_target(args, train_size)
    else:
        noisy_target = None
    
    print(ld_train.shape,ld_test.shape)
    valid_idx = random.sample(range(hd_test.shape[0]),40)
    train_data = DataClass(ld_train,hd_train,args.mode, args.semi_sup, args.supervision, noisy_target)
    test_data = DataClass(ld_test,hd_test,args.mode)
    valid_data = DataClass(ld_test[valid_idx],hd_test[valid_idx],args.mode)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=True)
        
    return train_dataloader, test_dataloader, valid_dataloader


    
class DataClass(Dataset):
    def __init__(self, ld, hd, mode, semi_sup=True, supervision=1.0, noisy_target=None):
        self.ld = ld
        self.hd = []
        self.label = [] 
        self.ref = hd
        
        if semi_sup and supervision!=1.0 and noisy_target is not None:
            sup_len = int(supervision*len(hd))
            if len(noisy_target)==1:
                noisy_target = noisy_target[0]
                self.hd[:sup_len] = hd[:sup_len]
                self.hd[sup_len:] = noisy_target[sup_len:]
                self.label[:sup_len] = [1]*(sup_len-1)
                self.label[sup_len:] = [0]*len(ld-sup_len+1)
                print("1 secondary taget used")
            else:
                self.hd[:sup_len] = hd[:sup_len]
                targets = []
                for i in range(len(noisy_target)):
                    targets.append(noisy_target[i][sup_len:])
                targets=np.array(targets).swapaxes(0,1)
                self.hd[sup_len:] = targets
                self.label[:sup_len] = [1]*(sup_len-1)
                self.label[sup_len:] = [0]*len(ld-sup_len+1)
                print("Multiple secondary tagets used")
        elif semi_sup and supervision!=1.0:
            sup_len = int(supervision*len(hd))
            self.hd[:sup_len] = hd[:sup_len]
            self.hd[sup_len:] = ld[sup_len:]
            self.label[:sup_len] = [1]*(sup_len-1)
            self.label[sup_len:] = [0]*len(ld-sup_len+1)
            print("same used")
        else:
            print("supervised")
            self.hd = hd
            self.label = [1]*len(hd)
            
    def __len__(self):
        return len(self.ld)

    def __getitem__(self, idx):
        if(len(self.hd[idx].shape)==4):
            stack = np.concatenate((np.expand_dims(self.ld[idx],0),self.hd[idx]),axis=0)
            samples = random.sample(range(self.hd[idx].shape[0]),2)
            ld = stack[samples[0]]
            hd = stack[samples[1]]
        else:
            if(self.label[idx]==1):
                ld = self.ld[idx]
                hd = self.hd[idx]
            else:
                if np.random.random()>0.5:
                    ld = self.ld[idx]
                    hd = self.hd[idx]
                else:
                    hd = self.ld[idx]
                    ld = self.hd[idx]
        sample = (ld,hd,self.ref[idx],idx,self.label[idx])
        return sample