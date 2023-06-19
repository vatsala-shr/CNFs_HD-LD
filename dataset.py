import glob
import os
import cv2
import torch
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from time import sleep
import torchvision
import numpy as np
    
class CT:
    def __init__(self, path = 'data/', transform = None, train = True, num_hd = 200, num_crap = 0, si_ld = False, 
                 noise_iter = 1, noise = True):
        random.seed(1)
        self.train = train
        self.transform = transform
        self.path = path
        self.list_crap = random.sample(range(200), num_crap)
        self.list_ld_pair = random.sample(range(200), 200 - num_hd)
        self.t = transforms.ToTensor()
        self.si_ld = si_ld
        self.noise_iter = noise_iter
        self.noise = noise
    
    def __len__(self):
        if self.train:
            return 200
        else:
            return 326

    def __getitem__(self, idx):
        if self.train:
            start = 0 + idx
        else:
            start = 500 + idx

        ct_hd = cv2.imread(self.path + f'ct/hd/{start}.png')[:, :, 0] 
        ct_ld = cv2.imread(self.path + f'ct/ld/{start}.png')[:, :, 0]
        ct_ld_out = cv2.imread(self.path + f'ct/ld_out/{start}.png')[:, :, 0] 
        pet_hd = cv2.imread(self.path + f'pet/hd/{start}.png')[:, :, 0] 
        pet_ld = cv2.imread(self.path + f'pet/ld/{start}.png')[:, :, 0]
        pet_ld_out = cv2.imread(self.path + f'pet/ld_out/{start}.png')[:, :, 0]
        mask = cv2.imread(self.path + f'ct/mask/{start}.png')[:, :, 0]
        
        if start in self.list_crap:
            # For added noise low dose
            if self.noise:
                for i in range(self.noise_iter):
                    noise = np.random.normal(loc = 0, scale = 1, size = ct_ld.shape).astype(np.uint8)
                    ct_ld = cv2.add(ct_ld, noise)
                    pet_ld = cv2.add(pet_ld, noise)
            else:
                # For out of distribution low dose
                ct_ld = cv2.imread(self.path + f'ct/ld_out/{start}.png')[:, :, 0]
                pet_ld = cv2.imread(self.path + f'pet/ld_out/{start}.png')[:, :, 0]

        if start in self.list_ld_pair:
            if self.si_ld:
                val = 'ld'
            else:
                val = 'ld1'
            ct_hd = cv2.imread(self.path + f'ct/{val}/{start}.png')[:, :, 0]
            pet_hd = cv2.imread(self.path + f'pet/{val}/{start}.png')[:, :, 0]

        ct_hd = self.t(ct_hd)
        ct_ld = self.t(ct_ld)
        pet_hd = self.t(pet_hd)
        pet_ld = self.t(pet_ld)
        mask = self.t(mask)
        ct_ld_out = self.t(ct_ld_out)
        pet_ld_out = self.t(pet_ld_out)
        mask = mask[:, 2:-2, 2:-2]
        # x = torch.concat([ct_hd, pet_hd, ct_ld, pet_ld], dim = 0)
        x = torch.concat([ct_hd, pet_hd, ct_ld, pet_ld, mask, ct_ld_out, pet_ld_out], dim = 0)
        if self.transform is not None:
            x = self.transform(x)
        
        return x


# transform = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                     transforms.RandomVerticalFlip()])
    
# train_set = CT(num_hd = int(1 * 200),
#                transform=None,
#                num_crap=0,
#                noise = True,
#                noise_iter = 0)
# test_set = CT(transform = transform,
#                     train = False)

# for i in range(train_set.__len__()):
#     x_prime = train_set.__getitem__(i).unsqueeze(1)
#     print(x_prime.shape)
#     # image = torchvision.utils.make_grid(x_prime, nrow = 2, padding = 2, pad_value = 128)
#     # torchvision.utils.save_image(image, f'train_data/{i}.png')
#     fig, ax = plt.subplots(1, 2, figsize = (30, 30))
#     ax[0].imshow(x_prime[0, 0, :, :], cmap = 'gray')
#     ax[1].imshow(x_prime[2, 0, :, :], cmap = 'gray')
#     ax[0].axis('off')
#     ax[1].axis('off')
#     ax[0].set_title('High Dose (CT)', fontsize = 30)
#     ax[1].set_title('Low Dose (CT)', fontsize = 30)
#     fig.subplots_adjust(wspace = -0.05)
#     plt.savefig(f'train_data/{i}_ct.png', bbox_inches = 'tight')
#     break