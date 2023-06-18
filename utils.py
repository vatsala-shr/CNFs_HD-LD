from tqdm import tqdm
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pdb
from pytorch_msssim import ssim as ssim1

@torch.no_grad()
def sample(net, origin_img, gray_img, device, sigma=0.6):
    B, C, W, H = origin_img.shape
    z = torch.zeros((B, C, W, H), device = device)
    # z = torch.randn((B, C, W, H), dtype=torch.float32, device=device) * sigma
    x, _ = net(z, gray_img, reverse=True)
    x = torch.sigmoid(x)
    return x

def evaluate_1c(net, loader, device, type = 'ct'):
    rrmse_val = list()
    psnr_val = list()
    ssim_val = list()

    for i in tqdm(loader, 'Evaluating:'):
        x_prime = i
        idx1, idx2 = get_idx(type)
        origin_img = x_prime[:, idx1, :, :].to(device)
        gray_img = x_prime[:, idx2, :, :].to(device)
        mask = x_prime[:, 4, :, :].unsqueeze(1).to(device)
        mask = torch.where(mask > 0, 1, 0)
        if len(origin_img.shape) < 4:
            origin_img = origin_img.unsqueeze(1)
        if len(gray_img.shape) < 4:
            gray_img = gray_img.unsqueeze(1)
        # Save samples and data
        images = sample(net, origin_img, gray_img, device)

        # samp_no = 2
        # for i in range(samp_no):
        #     images += sample(net, origin_img, gray_img, device)
        # images = images / samp_no

        images = images * mask
        r = F.mse_loss(origin_img, images).sqrt().item()
        p = psnr(origin_img.detach().cpu().numpy(), images.detach().cpu().numpy(), data_range = 1)
        
        min_val, max_val = -1024, 1500  
        images = (images * (max_val - min_val)) + min_val
        origin_img = (origin_img * (max_val - min_val)) + min_val
        t1 = images.squeeze(1).detach().cpu().numpy()
        t2 = origin_img.squeeze(1).detach().cpu().numpy()
        s = 0
        for idx in range(t1.shape[0]):
            _, smap = ssim(t1[idx],
                           t2[idx],
                           data_range = max_val - min_val, 
                           gaussian_weights=True,
                           sigma=4,
                           full=True)
            s += np.mean(smap)

        # s = ssim1(t1.unsqueeze(1).numpy(), t2.unsqueeze(1).numpy(), data_range = max_val - min_val)        
        ssim_val.append(s / t1.shape[0])
        # ssim_val.append(s.detach().cpu())
        rrmse_val.append(r)
        psnr_val.append(p)
    
    rrmse_val = np.array(rrmse_val)
    psnr_val = np.array(psnr_val)
    print(f'RRMSE : {np.mean(rrmse_val)}, PSNR : {np.mean(psnr_val)}, SSIM : {np.mean(ssim_val)}')
    return rrmse_val, psnr_val, ssim_val

def create_boxplot(labels, values, title, name):
    # Create a box plot for accuracy
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    val = [values[i] for i in range(values.shape[0])]
    bp = ax.boxplot(val, patch_artist=True, notch=True, vert=0)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for whisker in bp['whiskers']:
        whisker.set(color='#8c8c8c', linewidth=1.5, linestyle=":")

    for cap in bp['caps']:
        cap.set(color='#8c8c8c', linewidth=2)

    for median in bp['medians']:
        median.set(color='red', linewidth=3)

    for flier in bp['fliers']:
        flier.set(marker='D', color='#e7298a', alpha=0.5)

    for i in range(values.shape[0]):
        mean = np.mean(values[i])
        std = np.std(values[i])
        ax.text(mean, i + 1, '{:.3f}'.format(mean), va='center')
        # ax.text(mean - 0.2 * std, i + 1, 'Std: {:.2f}'.format(std * 100), va='center')

    ax.set_yticklabels(labels)
    # ax.set_ylabel('% of Supervision')
    ax.set_ylabel('Loss Function')
    # ax.set_ylabel('Conditional Input')
    # ax.set_ylabel('Separate Coupling Layer for Conditional Input')
    plt.title(title)
    plt.savefig(f'{name}.png', bbox_inches = 'tight')

def boxplot_helper_1c(net, loader, device, labels, type = 'ct'):
    rrmse_val = list()
    psnr_val = list()
    ssim_val = list()
    for i in labels:
        checkpoint = torch.load(f'ckpts/{type}/{i}_best.pth.tar', map_location = device)
        net.load_state_dict(checkpoint['net'])
        rrmse, psnr, ssim = evaluate_1c(net, loader, device)
        rrmse_val.append(rrmse)
        psnr_val.append(psnr)
        ssim_val.append(ssim)
    p = f'experiments/supervision/{type}/'
    os.makedirs(p, exist_ok=True)
    labels = [float(i) * 100 for i in labels]
    rrmse_val, psnr_val, ssim_val = np.array(rrmse_val), np.array(psnr_val), np.array(ssim_val)
    create_boxplot(labels, rrmse_val, f'{type.upper()}_RRMSE (Single Channel)', p + 'rrmse')
    create_boxplot(labels, psnr_val, f'{type.upper()}_PSNR (Single Channel)', p + 'psnr')
    create_boxplot(labels, ssim_val, f'{type.upper()}_SSIM (Single Channel)', p + 'ssim')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_idx(type):
    if type == 'ct':
        idx1 = 0
        idx2 = 2
    elif type == 'pet':
        idx1 = 1
        idx2 = 3
    elif type == 'ct_cond_pet':
        idx1 = 0
        idx2 = [2, 1, 3]
    elif type == 'pet_cond_ct':
        idx1 = 1
        idx2 = [3, 0, 2]
    elif type == 'ct_out':
        idx1 = 0
        idx2 = 5
    elif type == 'pet_out':
        idx1 = 1
        idx2 = 6

    return idx1, idx2