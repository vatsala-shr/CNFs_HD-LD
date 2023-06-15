import argparse
import torch
import random
import numpy as np
from dataset import CT
from torchvision import transforms
import torch.utils.data as data
from models import Glow
from tqdm import tqdm
from utils import sample, evaluate_1c, get_idx
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import os
import torch.nn.functional as F
from utils import boxplot_helper_1c, create_boxplot, count_parameters

def main(args):
    # Set up main device
    device = torch.device(f'cuda:{args.gpu_id}')

    # Set random seeds for same results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load the dataset
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()])
    train_set = CT(transform = transform,
                          num_hd = int(args.sup_ratio * 200),
                          num_crap = int(args.crap_ratio * 200),
                          si_ld=args.si_ld,
                          noise=args.noise,
                          noise_iter=args.noise_iter)
    test_set = CT(train = False)
    trainloader = data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=args.num_workers)
    testloader = data.DataLoader(test_set, batch_size=2, shuffle=False, num_workers=args.num_workers)

    print(f'Supervision Ratio : {args.sup_ratio}\nCrap Ratio : {args.crap_ratio}\nShape Parameter : {args.shape}\nNoise : {args.noise}\nNoise Iteration : {args.noise_iter}')


    # Model
    print('Building model..')
    net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps,
               mode=args.mode,
               inp_channel=args.inp_channel,
               cond_channel=args.cond_channel,
               cc = args.cc)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark
    
    # Loading the correct weights
    # path = f'ckpts/new_loss/{args.type}/{args.sup_ratio}_sl_best.pth.tar'
    path = f'ckpts/resnet/{args.type}/{args.sup_ratio}_best.pth.tar'
    # path = f'ckpts/robust/{args.type}/noise/{args.crap_ratio}/{args.noise_iter}/{args.shape}_best.pth.tar'
    checkpoint = torch.load(path, 
                            map_location = device)
    net.load_state_dict(checkpoint['net'])
    # print('Correct weights loaded!')

    rrmse = checkpoint['rrmse']
    psnr = checkpoint['psnr']
    ssim = checkpoint['ssim']
    print(f'RRMSE: {rrmse}, PSNR: {psnr}, SSIM: {ssim}')

    net.eval()
    rrmse, psnr, ssim = evaluate_1c(net, testloader, device, args.type)

    # # # Finding standard deviation across samples generated under certain sup_ratio
    # sup_ratio = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    # shape = [0.75, 1.0, 1.5, 2.0]

    # # Finding the standard deviation
    # for i in sup_ratio:
    #     std_dev(net, testloader, device, i, args.type, args.cc)

    # Visualizing the results
    # for i in sup_ratio:
    #     print(i)
    # result(net, testloader, device, args.sup_ratio, args.type)

    # shape = [0.5, 0.75, 1.0, 1.5, 2.0]
    # noise_iter = [1, 2, 4, 8, 16]
    # for j in noise_iter:
    #     # Checking the metrics related to different experiments
    #     rrmse_val = list()
    #     psnr_val = list()
    #     ssim_val = list()
    #     for i in shape:
    #         print(i)
    #         # Loading the correct weights
    #         checkpoint = torch.load(f'ckpts/robust/{args.type}/noise/{args.crap_ratio}/{j}/{i}_best.pth.tar', 
    #                                 map_location = device)
    #         print('Building model..')
    #         net = Glow(num_channels=args.num_channels,
    #                num_levels=args.num_levels,
    #                num_steps=args.num_steps,
    #                mode=args.mode,
    #                inp_channel=args.inp_channel,
    #                cond_channel=args.cond_channel,
    #                cc = args.cc)
    #         net = net.to(device)
    #         net.load_state_dict(checkpoint['net'])
    #         rrmse = checkpoint['rrmse']
    #         psnr = checkpoint['psnr']
    #         ssim = checkpoint['ssim']
    #         print(f'RRMSE: {rrmse}, PSNR: {psnr}, SSIM: {ssim}')
    #         print('Correct weights loaded!')


    #         # Evaluate the model
    #         net.eval()
    #         rrmse, psnr, ssim = evaluate_1c(net, testloader, device, args.type)
    #         rrmse_val.append(rrmse)
    #         psnr_val.append(psnr)
    #         ssim_val.append(ssim)

    #     rrmse_val, psnr_val, ssim_val = np.array(rrmse_val), np.array(psnr_val), np.array(ssim_val)
    #     p = f'experiments/robust/{args.type}/noise/{args.crap_ratio}/{j}/'
    #     os.makedirs(p, exist_ok=True)
    #     create_boxplot(shape, rrmse_val, f'RRMSE', p + 'rrmse')
    #     create_boxplot(shape, psnr_val, f'PSNR', p + 'psnr')
    #     create_boxplot(shape, ssim_val, f'SSIM', p + 'ssim')


@torch.no_grad()
def result(net, loader, device, sup_ratio=1.0, type='ct'):
    # Loading the correct weights
    path = f'ckpts/resnet/{type}/{sup_ratio}_best.pth.tar'
    # path = f'ckpts/new_loss/{type}/{sup_ratio}_sl_best.pth.tar'
    checkpoint = torch.load(path, 
                            map_location = device)
    net.load_state_dict(checkpoint['net'])
    print('Correct weights loaded!')
    net.eval()
    idx1, idx2 = get_idx(type)
    print(f'idx1 : {idx1}, idx2 : {idx2}')

    # Path to save results
    path = f'experiments/new_loss/{type}/{sup_ratio}/'
    # path = f'experiments/new_loss/{type}/{sup_ratio}_sl/'
    os.makedirs(path, exist_ok = True)

    # Calculation
    for c, i in tqdm(enumerate(loader)):
        gt_hd = i[:, idx1, :, :].to(device)
        ld = i[:, idx2, :, :].to(device)

        if len(gt_hd.shape) < 4:
            gt_hd = gt_hd.unsqueeze(1)
        if len(ld.shape) < 4:
            ld = ld.unsqueeze(1)
        
        # Sample from the model
        pred_hd = sample(net, gt_hd, ld, device)
        pred_hd = pred_hd * i[:, 4, :, :].unsqueeze(1).to(device)
        gt_hd = gt_hd.detach().cpu()
        ld = ld.detach().cpu()
        pred_hd = pred_hd.detach().cpu()
        # x = torch.concat([ld, gt_hd, pred_hd], dim = 1)
        x = torch.concat([ld, gt_hd, pred_hd, torch.abs(gt_hd - pred_hd)], dim = 1)
        plot1(x, sup_ratio, file = f'{path}{c}.png')
        if c == 20:
            break


@torch.no_grad()
def std_dev(net, loader, device, sup_ratio = 1.0, type = 'ct', cc = True):
    # Loading the correct weights
    checkpoint = torch.load(f'ckpts/resnet/{type}/{sup_ratio}_best.pth.tar', 
                            map_location = device)
    net.load_state_dict(checkpoint['net'])
    print('Correct weights loaded!')
    net.eval()
    idx1, idx2 = get_idx(type)

    # Path to save results
    path = f'experiments/std_across_sup/{type}/{sup_ratio}/'
    os.makedirs(path, exist_ok = True)

    # Calculation
    for c, i in tqdm(enumerate(loader)):
        gt_hd = i[:, idx1, :, :].to(device)
        ld = i[:, idx2, :, :].to(device)

        if len(gt_hd.shape) < 4:
            gt_hd = gt_hd.unsqueeze(1)
        if len(ld.shape) < 4:
            ld = ld.unsqueeze(1)
        
        # Find the sample hd conditioned on ld
        pred_hd = sample(net, gt_hd, ld, device)
        for i in tqdm(range(100), 'Generating Samples:'):
            new_sample = sample(net, gt_hd, ld, device)
            pred_hd = torch.concat([pred_hd, new_sample], dim = 1)
        
        std = torch.std(pred_hd, dim = 1).unsqueeze(1).detach().cpu()
        mean = torch.mean(pred_hd, dim = 1).unsqueeze(1).detach().cpu()
        gt_hd = gt_hd.detach().cpu()
        print('Standard Deviation Calculated!')
        x = torch.concat([gt_hd, mean, std], dim = 1)
        plot(x, sup_ratio, file = f'{path}{c}.png')
        if c == 5:
            break

def plot(x, sup_ratio, file = 'testing.png'):
    imgs = x.shape[0]
    batch = x.shape[1]
    labels = ['Ground Truth', 'Mean', 'Standard Deviation']
    fig, ax = plt.subplots(imgs, batch, figsize = (30, 30))
    fig.subplots_adjust(wspace = 0.01, hspace = -0.48)
    for i in range(batch):
        for j in range(imgs):
            if i != 2:
                ax[j, i].imshow(x[j, i, :, :], cmap = 'gray')
            else:
                ax[j, i].imshow(x[j, i, :, :], cmap = 'jet')
                im = ax[j, i].imshow(x[j, i, :, :], cmap = 'jet', 
                         vmin = x[j, i, :, :].min(),
                         vmax = np.percentile(x[j, i, :].numpy(), 99.2))
            
            if j == 0:
                ax[j, i].set_title(labels[i], fontsize = 40)

            ax[j, i].axis('off')

    title = plt.title(f'Standard Deviation for {int(sup_ratio * 100)}% Supervision',
              loc = 'center', x = -0.5, y = 1.95)
    title.set_fontsize(40) 
    cbar = plt.colorbar(im, ax=ax, orientation = 'horizontal', 
                 pad = 0.01, aspect = 100)
    cbar.ax.tick_params(labelsize=30)
    plt.savefig(file, bbox_inches = 'tight')
    plt.close()

def plot1(x, sup_ratio, file = 'testing.png'):
    imgs = x.shape[0]
    batch = x.shape[1]
    labels = ['Low Dose', 'Ground Truth', 'Predicted', 'Absolute Residue Value']
    fig, ax = plt.subplots(imgs, batch, figsize = (40, 30))
    fig.subplots_adjust(wspace = 0.01, hspace = -0.48)
    for i in range(batch):
        for j in range(imgs):
            if i != 3:
                ax[j, i].imshow(x[j, i, :, :], cmap = 'gray')
            else:
                # print(x[j, i, :, :].min(),  np.percentile(x[j, i, :].numpy(), 99.2))
                im = ax[j, i].imshow(x[j, i, :, :], cmap = 'jet')
                        #  vmin = 0,
                        #  vmax = 0.04)
                        #  vmax = np.percentile(x[j, i, :].numpy(), 99.9))
            ax[j, i].axis('off')
            if j == 0:
                ax[j, i].set_title(labels[i], fontsize = 40)

    title = plt.title(f'Results for {int(sup_ratio * 100)}% Supervision',
              loc = 'center', x = -1, y = 1.96)
    title.set_fontsize(40) 
    cbar = plt.colorbar(im, ax=ax, orientation = 'horizontal', 
                 pad = 0.01, aspect = 100)
    cbar.ax.tick_params(labelsize=30)
    plt.savefig(file, bbox_inches = 'tight')
    plt.close()

if __name__ == '__main__':
    def str2bool(s):
        return s.lower().startswith('t')
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_channels', type = int, default = 128)
    parser.add_argument('--num_levels', type = int, default = 4)
    parser.add_argument('--num_steps', type = int, default = 8)
    parser.add_argument('--gpu_id', type = int, default = 0)
    parser.add_argument('--sup_ratio', type = float, default = 1.0)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--num_workers', type = int, default = 8)
    parser.add_argument('--type', type = str, default = 'ct')
    parser.add_argument('--cc', type=str2bool, default=False)
    parser.add_argument('--inp_channel', type = int, default=1)
    parser.add_argument('--cond_channel', type=int, default=1)
    parser.add_argument('--shape', type=float, default=1)
    parser.add_argument('--mode', default="sketch", choices=['gray', 'sketch'])
    parser.add_argument('--noise_iter', default=1, type=int)
    parser.add_argument('--crap_ratio', default=0.0, type=float)
    parser.add_argument('--noise', default=False, type=bool)
    parser.add_argument('--si_ld', type=bool, default=False)
    main(parser.parse_args())