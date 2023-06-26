import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import util
from torchvision import transforms
from dataset import CT
import pickle
from torch.utils.data import Subset
import torch.nn.functional as F
from time import sleep
from utils import evaluate_1c
import pdb
# from pytorch_msssim import ssim as ssim1

from models import Glow
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import count_parameters, sample, get_idx

# from skimage.metrics import structural_similarity as ssim
import numpy as np
import pickle
# import kornia.losses as loss
from utils import perceptual_loss
from models.glow.coupling import UNet1

def main(args):
    # Set up main device and scale batch size
    device = torch.device(f'cuda:{args.gpu_id}')
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()])
    print(f'Supervision Ratio : {args.sup_ratio}\nCrap Ratio : {args.crap_ratio}\nShape Parameter : {args.shape}\nNoise : {args.noise}\nNoise Iteration : {args.noise_iter}')
    
    train_set = CT(transform = transform,
                          num_hd = int(args.sup_ratio * 200),
                          num_crap = int(args.crap_ratio * 200),
                          si_ld=args.si_ld,
                          noise=args.noise,
                          noise_iter=args.noise_iter)
    test_set = CT(train = False)
    trainloader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers)
    testloader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

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

    

    unet = UNet1(inp_channels=1, op_channels=1)
    unet = unet.to(device)
    unet_weights = torch.load('ckpts/unet/best.pth', map_location = device)
    unet.load_state_dict(unet_weights)

    # # Different paths for different experiments
    # path = f'ckpts/shape/{args.type}/{args.shape}/{args.sup_ratio}_best.pth.tar'
    # path = f'ckpts/robust/{args.type}/out_dist/{args.crap_ratio}/{args.shape}_best.pth.tar'
    # path = f'ckpts/robust/{args.type}/noise/{args.crap_ratio}/{args.noise_iter}/{args.shape}_best.pth.tar'
    path = f'ckpts/adv/{args.type}/{args.sup_ratio}_{args.ext}_out.pth.tar'
   
    start_epoch = 0
    global best_ssim
    global global_step
    global best_epoch
    if args.resume and os.path.exists(path):
        # Load checkpoint.
        print(path)
        checkpoint = torch.load(path, map_location = device)
        net.load_state_dict(checkpoint['net'])
        best_ssim = checkpoint['ssim']
        start_epoch = checkpoint['epoch']
        best_epoch = start_epoch
        print(f'Best SSIM : {best_ssim}, Start Epoch : {start_epoch}')
        global_step = start_epoch * len(train_set)
    else:
         best_epoch = 0
         best_ssim = 0

    loss_fn = util.NLLLoss().to(device)
    # loss_fn = util.NLLLoss(shape = args.shape, device = device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))
    print(f'Number of parameters : {count_parameters(net)}')

    epoch = start_epoch
    c = 0
    history = []
    while epoch <= args.num_epochs:
        c += 1
        train(epoch, net, trainloader, device, optimizer, scheduler,
              loss_fn, type = args.type, args = args, model=unet)
        if test(epoch, net, testloader, device, args, path, history):
            if os.path.exists(path):  
                checkpoint = torch.load(path, map_location = device)
                net.load_state_dict(checkpoint['net'])
                best_ssim = checkpoint['ssim']
                best_epoch = checkpoint['epoch']
                epoch = best_epoch
                print('Loaded previous model...')
            else:
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
                best_ssim = 0
                best_epoch = start_epoch
                epoch = start_epoch
                print('Initialized new model!')

            optimizer = optim.Adam(net.parameters(), lr=args.lr)
            scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))
        else:
            epoch += 1

        # Early Stopping
        print(f'Current Epoch - Best Epoch : {(epoch - best_epoch)}, Epochs Completed / Total Epochs : {c}/{args.num_epochs}')
        # if (epoch - best_epoch) >= 100:
        #     print('Early Stopping...')
        #     print(f"Best SSIM : {best_ssim}")
        #     break

        if c == args.num_epochs:
            print('Sufficient Epoch Completed!')
            print(f'Best SSIM : {best_ssim}')
            break
        # print(history)

        os.makedirs(f'history/{args.type}', exist_ok=True)
        file = open(f'history/{args.type}/{args.sup_ratio}_{args.ext}.pkl', 'wb')
        pickle.dump(history, file)

@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm = -1, type = 'ct', args = None, model = None, epsilon = 5e-4):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    latent_loss_m = util.AverageMeter()
    spatial_loss_m = util.AverageMeter()
    idx1, idx2 = get_idx(type)
    smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)

    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for i, x_prime in enumerate(trainloader):
            x = x_prime[:, idx1, :, :]
            cond_x = x_prime[:, idx2, :, :]
            mask = x_prime[:, 4, :, :].unsqueeze(1).to(device)
            mask = torch.where(mask > 0, 1, 0)
            if len(x.shape) < 4:
                x = x.unsqueeze(1)
            if len(cond_x.shape) < 4:
                cond_x = cond_x.unsqueeze(1)
            x, cond_x = x.to(device), cond_x.to(device)
            cond_x.requires_grad = True
            z, sldj = net(x, cond_x, reverse=False)

            if args.ext == 'll+sl_pl' or args.ext == 'll+sl_l1':
                sl_name = args.ext.split('_')[1]
                optimizer.zero_grad()
                latent_loss = loss_fn(z, sldj)
                new_z = torch.randn(x.shape, dtype=torch.float32, device=device) * 0.6
                rec_x, sldj = net(new_z, cond_x, reverse=True)
                rec_x = mask * torch.sigmoid(rec_x)
                if sl_name == 'pl':
                    spatial_loss = perceptual_loss(rec_x, x, model, smooth_l1_loss)
                elif sl_name == 'l1':
                    spatial_loss = smooth_l1_loss(rec_x, x)

                loss = latent_loss + spatial_loss
                loss.backward()
                if max_grad_norm > 0:
                    util.clip_grad_norm(optimizer, max_grad_norm)
                optimizer.step()
                scheduler.step(global_step)

                latent_loss_m.update(latent_loss.item(), x.size(0))
                spatial_loss_m.update(spatial_loss.item(), x.size(0))
                progress_bar.set_postfix(bpd=util.bits_per_dim(x, latent_loss_m.avg),
                                         pl = spatial_loss_m.avg)
                progress_bar.update(x.size(0))
                global_step += x.size(0)

                # Adversarial Examples Training

                # Calculating FGSM
                cond_x = cond_x + (epsilon * cond_x.grad)
                cond_x = cond_x - cond_x.min() / cond_x.max() - cond_x.min()
                cond_x[cond_x.isnan()] = 0

                # Feeding to the network and calculating loss
                z, sldj = net(x, cond_x, reverse=False)
                optimizer.zero_grad()
                latent_loss = loss_fn(z, sldj)
                new_z = torch.randn(x.shape, dtype=torch.float32, device=device) * 0.6
                rec_x, sldj = net(new_z, cond_x, reverse=True)
                rec_x = mask * torch.sigmoid(rec_x)
                if sl_name == 'pl':
                    spatial_loss = perceptual_loss(rec_x, x, model, smooth_l1_loss)
                elif sl_name == 'l1':
                    spatial_loss = smooth_l1_loss(rec_x, x)
                loss = latent_loss + spatial_loss

                # Backpropagation
                loss.backward()
                if max_grad_norm > 0:
                    util.clip_grad_norm(optimizer, max_grad_norm)
                optimizer.step()
                scheduler.step(global_step)

                # Updating the progress bar
                latent_loss_m.update(latent_loss.item(), x.size(0))
                spatial_loss_m.update(spatial_loss.item(), x.size(0))
                progress_bar.set_postfix(bpd=util.bits_per_dim(x, latent_loss_m.avg),
                                         pl = spatial_loss_m.avg)
                progress_bar.update(x.size(0))
                global_step += x.size(0)

            # elif args.ext == 'll_then_sl':
            #     optimizer.zero_grad()
            #     latent_loss = loss_fn(z, sldj)
            #     latent_loss.backward()
            #     if max_grad_norm > 0:
            #         util.clip_grad_norm(optimizer, max_grad_norm)
            #     optimizer.step()
            #     scheduler.step(global_step)

            #     new_z = torch.randn(x.shape, dtype=torch.float32, device=device) * 0.6
            #     rec_x, sldj = net(new_z, cond_x, reverse=True)
            #     rec_x = mask * torch.sigmoid(rec_x)
            #     optimizer.zero_grad()
            #     # l1_loss = smooth_l1_loss(rec_x, x)
            #     # ssim_loss = 1 - ssim1(rec_x, x, data_range = 1)
            #     # spatial_loss = l1_loss + ssim_loss
            #     spatial_loss = perceptual_loss(rec_x, x, model, smooth_l1_loss)
            #     spatial_loss.backward()
            #     if max_grad_norm > 0:
            #         util.clip_grad_norm(optimizer, max_grad_norm)
            #     optimizer.step()
            #     scheduler.step(global_step)
            #     print(f'll:{latent_loss}, sl:{spatial_loss}')
            #     latent_loss_m.update(latent_loss.item(), x.size(0))
            #     # l1_loss_m.update(l1_loss.item(), x.size(0))
            #     # ssim_loss_m.update(ssim_loss.item(), x.size(0))
            #     perceptual_loss_m.update(spatial_loss.item(), x.size(0))
            #     progress_bar.set_postfix(bpd=util.bits_per_dim(x, latent_loss_m.avg),
            #                              pl=perceptual_loss_m.avg)
            #                             # ssim=ssim_loss_m.avg,
            #                             # l1=l1_loss_m.avg)
            #     progress_bar.update(x.size(0))
            #     global_step += x.size(0)

            elif args.ext == 'll':
                optimizer.zero_grad()
                latent_loss = loss_fn(z, sldj)
                latent_loss.backward()
                if max_grad_norm > 0:
                    util.clip_grad_norm(optimizer, max_grad_norm)
                optimizer.step()
                scheduler.step(global_step)

                latent_loss_m.update(latent_loss.item(), x.size(0))
                progress_bar.set_postfix(nll=latent_loss_m.avg,
                                        bpd=util.bits_per_dim(x, latent_loss_m.avg),
                                        lr=optimizer.param_groups[0]['lr'])
                progress_bar.update(x.size(0))
                global_step += x.size(0)

                # Adversarial Examples Training
                cond_x = cond_x + (epsilon * cond_x.grad)
                cond_x = cond_x - cond_x.min() / cond_x.max() - cond_x.min()
                cond_x[cond_x.isnan()] = 0
                z, sldj = net(x, cond_x, reverse=False)
                optimizer.zero_grad()
                latent_loss = loss_fn(z, sldj)
                latent_loss.backward()
                if max_grad_norm > 0:
                    util.clip_grad_norm(optimizer, max_grad_norm)
                optimizer.step()
                scheduler.step(global_step)
                
        
                latent_loss_m.update(latent_loss.item(), x.size(0))
                progress_bar.set_postfix(nll=latent_loss_m.avg,
                                        bpd=util.bits_per_dim(x, latent_loss_m.avg),
                                        lr=optimizer.param_groups[0]['lr'])
                progress_bar.update(x.size(0))
                global_step += x.size(0)



@torch.no_grad()
def test(epoch, net, testloader, device, args, path, history = []):
    global best_ssim
    global best_epoch
    net.eval()

    rrmse_val, psnr_val, ssim_val = evaluate_1c(net, testloader, device, args.type)
    ssim = np.mean(ssim_val)
    flag = True
    history.append(ssim)

    # Save checkpoint
    if torch.isnan(torch.tensor(ssim)):
        return True

    if flag and ssim > best_ssim:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'ssim': ssim,
            'rrmse': np.mean(rrmse_val),
            'psnr': np.mean(psnr_val),
            'epoch': epoch,
        }
        path1 = '/'.join(path.split('/')[:-1])
        os.makedirs(path1, exist_ok=True)
        torch.save(state, path)        
        best_ssim = ssim
        best_epoch = epoch

    return False
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_id', default=6, type=int, help='ID of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad _norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=128, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=4, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=8, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=True, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--mode', default="sketch", choices=['gray', 'sketch'])
    parser.add_argument('--sup_ratio', default = 0.0, type = float)
    parser.add_argument('--type', type = str, default = 'ct')
    parser.add_argument('--inp_channel', type=int, default=1)
    parser.add_argument('--cond_channel', type=int, default=1)
    parser.add_argument('--cc', type = str2bool, default = False)
    parser.add_argument('--shape', type = float, default=2)
    parser.add_argument('--crap_ratio', type = float, default=0.0)
    parser.add_argument('--si_ld', type = bool, default = False)
    parser.add_argument('--noise_iter', default=1, type=int)
    parser.add_argument('--noise', default = False, type=bool)
    parser.add_argument('--ext', default = 'll', type=str)
    best_loss = float('inf')
    best_ssim = 0
    best_epoch = 0
    global_step = 0

    main(parser.parse_args())