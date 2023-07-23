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
from utils import *
random.seed(0)

new_range=0
def set_new_range(num):
    global new_range
    new_range=num
    
def get_trunc_vals(mode):
    if new_range==0:
        if mode=='ct':
            trunc_min = -1024
            trunc_max = 320 #760
        elif mode =='pet':
            trunc_min = -8000
            trunc_max = 15000
    elif new_range==1:
        if mode=='ct':
            trunc_min = -1024
            trunc_max = 320 #760
        elif mode =='pet':
            trunc_min = -2000
            trunc_max = 20000
    elif new_range==2:
        if mode=='ct':
            trunc_min = -1024
            trunc_max = 320 #760
        elif mode =='pet':
            trunc_min = 0
            trunc_max = 20000
    return (trunc_min,trunc_max)

def get_standard_range(mode):
    if new_range==0:
        if mode=='ct':
            min_val = -1024
            max_val = 3072
        elif mode =='pet':
            min_val = -8000
            max_val = 15000
    elif new_range==1:
        if mode=='ct':
            min_val = -1024
            max_val = 1500
        elif mode =='pet':
            min_val = -2000
            max_val = 46000
    elif new_range==2:
        if mode=='ct':
            min_val = -1024
            max_val = 1500
        elif mode =='pet':
            min_val = 0
            max_val = 46000
     
    return (min_val,max_val)

def unnormalize_trunc(data, mode, trunc=True):
    min_val,max_val = get_standard_range(mode)
        
    data = min_val + (data * (max_val-min_val))
    
    if trunc:
        trunc_min, trunc_max = get_trunc_vals(mode)
        data[data <= trunc_min] = trunc_min
        data[data >= trunc_max] = trunc_max
    return data

#####################################################################################################

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
        
    hd = hd*65535
    ld = ld*65535
    hd[hd>65535]=65535
    ld[ld>65535]=65535
    hd=hd.astype('uint16').astype('float32')
    ld=ld.astype('uint16').astype('float32')
    hd=hd/65535
    ld=ld/65535
    
    train_size = args.train_size if args.semi_sup else int(args.supervision*args.train_size) #smaller train size when fully supervised 
    
    hd_train = hd[:train_size]
    ld_train = ld[:train_size]
    hd_test = hd[500:]
    ld_test = ld[500:]
        
    if args.semi_sup and args.supervision<1.0 and args.secondary_noisy:
        noisy_target = get_noisy_target(args, train_size)
        noisy_train = noisy_target[:train_size]
        noisy_test = noisy_target[500:]
        noisy_valid = noisy_target[valid_idx]
    else:
        noisy_train = noisy_test = noisy_valid = None
    
    print(ld_train.shape,ld_test.shape)
    valid_idx = random.sample(range(hd_test.shape[0]),40)
    train_data = DataClass(ld_train,hd_train,args.mode, args.semi_sup, args.supervision, noisy_train, train=True)
    test_data = DataClass(ld_test,hd_test,args.mode,args.semi_sup, args.supervision, noisy_test, train=False)
    valid_data = DataClass(ld_test[valid_idx],hd_test[valid_idx],args.mode, args.semi_sup, args.supervision, noisy_valid, train=False)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=True)
        
    return train_dataloader, test_dataloader, valid_dataloader


def get_noisy_target(args,train_size):
    data_path = '../../Pet_CT/data_masked/'
    if args.new_range == 1:
        data_path = data_path+'new_range/'
    elif args.new_range == 2:
        data_path = data_path+'new_range2/'
        
    if args.blur_mode=='petct':
        file_name = ['blur_pet','blur_ct']
    elif args.blur_mode=='pet':
        file_name = ['blur_pet','ct']
    elif args.blur_mode is None:
        file_name = ['pet','ct']
    else:
        print("INVALID blur mode")
    
    file_name = file_name[0] if args.mode=='pet' else file_name
    file_name = file_name[1] if args.mode=='ct' else file_name
        
    if isinstance(file_name, list):
        targ1 = nib.load(os.path.join(data_path,'data_ld',file_name[0]+'_'+str(args.noise_level[0])+'_1.nii.gz')).get_fdata().astype('float32')
        targ2 = nib.load(os.path.join(data_path,'data_ld',file_name[1]+'_'+str(args.noise_level[1])+'_1.nii.gz')).get_fdata().astype('float32')
        targ = np.stack((targ1,targ2),axis=1)
    else:
        targ = nib.load(os.path.join(data_path,'data_ld',file_name+'_'+str(args.noise_level[0])+'_1.nii.gz')).get_fdata().astype('float32')
        targ = np.expand_dims(targ,1)
        
    targ = targ*65535
    targ[targ>65535]=65535
    targ=targ.astype('uint16').astype('float32')
    targ=targ/65535
        
    return targ

    
class DataClass(Dataset):
    def __init__(self, ld, hd, mode, semi_sup=True, supervision=1.0, noisy_target=None, train=False):
        self.ld = ld
        self.hd = []
        self.label = [] 
        self.ref = hd
        
        if semi_sup and noisy_target is not None and train:
            sup_len = int(supervision*len(hd))
            self.hd[:sup_len] = hd[:sup_len]
            self.hd[sup_len:] = noisy_target[sup_len:]
            self.label[:sup_len] = [1]*(sup_len-1)
            self.label[sup_len:] = [0]*len(ld-sup_len+1)
            print("1 secondary taget used")
            
        elif semi_sup and train:
            sup_len = int(supervision*len(hd))
            self.hd[:sup_len] = hd[:sup_len]
            self.hd[sup_len:] = ld[sup_len:]
            self.label[:sup_len] = [1]*(sup_len-1)
            self.label[sup_len:] = [0]*len(ld-sup_len+1)
            print("same used")
        elif train==False:
            if noisy_target is not None:
                self.hd = noisy_target
                self.label = [0]*len(hd)
            else:
                self.hd = self.ld
                self.label = [0]*len(hd)
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

###########################################################################################

def get_masked_img(img,mask):
    # Add padding to make image same size as mask
    img = np.pad(img, 2, 'constant')
    
    masked = np.multiply(img, mask)
    
    # Return image after removing padding
    return masked[2:-2,2:-2]

def get_square_img(img,value=0):
    if(img.shape[0]<img.shape[1]):
        pad_len = int(img.shape[1]-img.shape[0])
        img_sq = np.pad(img,((pad_len//2,pad_len-(pad_len//2)),(0,0)),constant_values=(value,))
    elif (img.shape[0]>img.shape[1]):
        pad_len = int(img.shape[0]-img.shape[1])
        img_sq = np.pad(img,((0,0),(pad_len//2,pad_len-(pad_len//2))),constant_values=(value,))
    else:
        img_sq = img
    return img_sq

def compute_rmse(tensor1, tensor2):
#     rmse = np.sqrt(np.sum((tensor1-tensor2)**2)/np.sum(mask))
    return np.linalg.norm(tensor1-tensor2)/np.linalg.norm(tensor2)

def compute_ssim(tensor1, tensor2, data_range, mask):
    _, smap = ssim(tensor1,tensor2,data_range=data_range,gaussian_weights=True,sigma=4, full=True)
    smap_masked = np.multiply(np.pad(smap, 2, 'constant'),mask)
    return np.sum(smap_masked)/np.sum(mask)

def compute_errors(tensor1, tensor2, mask, limits, mode, tight=False):
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.detach().numpy()
        tensor2 = tensor2.detach().numpy()
        
    tensor1 = tensor1.reshape(tensor1.shape[-2:])
    tensor2 = tensor2.reshape(tensor2.shape[-2:])
    
    pred = get_masked_img(tensor1,mask)
    orig = get_masked_img(tensor2,mask)
    
    if tight==False:
        pred_img = get_square_img(np.pad(pred, 2, 'constant')[limits[0]:limits[1],limits[2]:limits[3]])
    else:
        pred_img = np.pad(pred, 2, 'constant')[limits[0]:limits[1],limits[2]:limits[3]]
        
    pred_img = unnormalize_trunc(pred_img,mode, trunc=True)
    pred = unnormalize_trunc(pred,mode, False)
    orig = unnormalize_trunc(orig,mode, False)

    min_val,max_val = get_standard_range(mode)
    data_range = max_val-min_val

    return (pred_img,compute_rmse(pred,orig),compute_ssim(pred,orig,data_range,mask),0)

##################################################################################################
def show(img, vmin=0.0, vmax=1.0, cmap='grey'):
    if isinstance(img, torch.Tensor):
        img = img.detach().numpy()#np.array(img)
    img = np.clip(img,vmin,vmax)
    img = img.reshape(img.shape[-2],img.shape[-1])
    img = (img - vmin)/(vmax-vmin)
    if cmap=='jet':
        img = Image.fromarray(np.uint8(cm.jet(img)*255))
    elif cmap=='hot':
        img = Image.fromarray(np.uint8(cm.hot(img)*255))
    else:
        img = Image.fromarray(np.uint8(img*255))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    display(img)

def save_fig(pred,errs,idx,save_path,mode):
    trunc_min, trunc_max = get_trunc_vals(mode)
    pred = np.clip(pred,trunc_min, trunc_max)
    pred = np.flipud(pred)
    f,ax = plt.subplots(1,1,figsize=(10, 10))
    cmap = plt.cm.gray if mode =='ct' else plt.cm.hot
    ax.imshow(pred, cmap=cmap, vmin=trunc_min, vmax=trunc_max)
    ax.set_xlabel("RMSE: {:.4f} SSIM: {:.4f}".format(errs[0],errs[1]), fontsize=20)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    
    f.savefig(os.path.join(save_path, mode+'_{}.png'.format(idx)),bbox_inches = 'tight')
    plt.close()
    
def create_save_path(args):
    text2 = 'semi' if args.semi_sup else 'full'
    text3 = str(args.noise_level[0])+'-'+str(args.noise_level[1]) if args.mode=='petct' else str(args.noise_level[0])
    text5 = 'same_' if args.secondary_noisy==0 else ''

    save_path = args.path+args.mode+'_'+text5+text2+'_'+text3+'_'+str(args.supervision)
    save_path_fig = os.path.join(save_path,'fig')
    print(save_path)
    if args.save and not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path_fig)
        print('Create path : {}'.format(save_path))
        
    args.save_path = save_path
    args.save_path_fig = save_path_fig
        
    return args

#####################################################################################################

def test_model(args,model,test_dataloader):
    with torch.no_grad():
        contours = pickle.load(open('../../Pet_CT/data_masked/contours.pkl','rb'))
        model.eval()
        errors = []
        i=0
        preds = []
        with tqdm(total=326) as pbar:
            for data in test_dataloader:
                x = data[0]
                cond_x = data[1]
                ref = data[2]
                mask = contours[i][1]
                limits = contours[i][0]
                
#                 show(x)
#                 show(cond_x)
#                 input()

                if torch.cuda.is_available():
                    x = x.to(args.device)
                    cond_x = cond_x.to(args.device)
                    
                pred = sample(model, x, cond_x, args.device).cpu()
                x = x.cpu()
                
                if args.mode == 'pet' or args.mode == 'ct':
                    pred_img,rmse_img,ssim_img, qilv_img = compute_errors(pred,ref,mask,limits,args.mode)
                    errors.append((rmse_img,ssim_img,qilv_img))

                    err_vals=(rmse_img,ssim_img,qilv_img)
                    if args.save:
                        save_fig(pred_img,err_vals,data[3].item(),args.save_path_fig,args.mode)
                else:
                    if save_nii:
                        preds.append(pred.detach().cpu().numpy())
                    pred_pet,rmse_pet,ssim_pet, qilv_pet = compute_errors(pred[:,0],ref[:,0],mask,limits,'pet')
                    err_vals=(rmse_pet,ssim_pet, qilv_pet)
                    if args.save:
                        save_fig(pred_pet,err_vals,data[3].item(),args.save_path_fig,'pet') 

                    pred_ct,rmse_ct,ssim_ct, qilv_ct = compute_errors(pred[:,1],ref[:,1],mask,limits,'ct')
                    err_vals=(rmse_ct,ssim_ct, qilv_ct)
                    if args.save:
                        save_fig(pred_ct,err_vals,data[3].item(),args.save_path_fig,'ct') 

                    errors.append((rmse_pet,ssim_pet, qilv_pet,rmse_ct,ssim_ct, qilv_ct))

                pbar.update(1)
                i=i+1
    return errors