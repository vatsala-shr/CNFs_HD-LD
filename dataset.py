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

        # print((ct_hd == ct_ld).all())
        ct_hd = self.t(ct_hd)
        ct_ld = self.t(ct_ld)
        pet_hd = self.t(pet_hd)
        pet_ld = self.t(pet_ld)
        mask = self.t(mask)
        ct_ld_out = self.t(ct_ld_out)
        pet_ld_out = self.t(pet_ld_out)
        mask = mask[:, 2:-2, 2:-2]
        x = torch.concat([ct_hd, pet_hd, ct_ld, pet_ld, mask, ct_ld_out, pet_ld_out], dim = 0)

        if self.transform is not None:
            x = self.transform(x)
        
        return x


# transform = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                     transforms.RandomVerticalFlip()])
    
# train_set = CT(num_hd = int(1 * 200),
#                transform=transform,
#                num_crap=200,
#                noise = True,
#                noise_iter = 0)
# test_set = CT(transform = transform,
#                     train = False)

# for i in range(train_set.__len__()):
#     x_prime = train_set.__getitem__(i).unsqueeze(1)
#     print(x_prime.shape)
#     print(x_prime[4, :, :].unique())
#     image = torchvision.utils.make_grid(x_prime, nrow = 5, padding = 2, pad_value = 128)
#     torchvision.utils.save_image(image, f'testing.png')
#     sleep(2)