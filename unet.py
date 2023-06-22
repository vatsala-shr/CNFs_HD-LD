from models.glow.coupling import UNet1
import torch
from dataset import CT
import argparse
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
from utils import get_idx
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import os


parser = argparse.ArgumentParser(description='PyTorch Glow')
parser.add_argument('--sup_ratio', default=1, type=float)
parser.add_argument('--crap_ratio', default=0, type=float)
parser.add_argument('--si_ld', default=False, type=bool)
parser.add_argument('--noise', default=False, type=bool)
parser.add_argument('--noise_iter', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--type', default='ct', type=str)
args = parser.parse_args()

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()])
train_set = CT(transform = transform,
                          num_hd = int(args.sup_ratio * 200),
                          num_crap = int(args.crap_ratio * 200),
                          si_ld=args.si_ld,
                          noise=args.noise,
                          noise_iter=args.noise_iter)
test_set = CT(transform = transform,
              train = False)

combined_dataset = ConcatDataset([train_set, test_set])

# trainloader = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers)
# testloader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
trainloader = data.DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=args.num_workers)
device = torch.device(f"cuda:{args.gpu_id}")

# Model
print('Building model..')
net = UNet1(inp_channels=1, op_channels=1)
net = net.to(device)


# Define loss function
criterion = torch.nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(net.parameters())

# Set number of epochs
num_epochs = 200

# Training loop
prev_best = 1000000
for epoch in range(num_epochs):
    running_loss = 0.0

    # Iterate over the training dataset
    for i, data in tqdm(enumerate(trainloader, 0)):
        # Get inputs and move to the GPU
        idx1, idx2 = get_idx(args.type)
        target = data[:, idx1, :, :]
        input = data[:, idx1, :, :]
        mask = data[:, 4, :, :].unsqueeze(1).to(device)
        mask = torch.where(mask > 0, 1, 0)
        if len(target.shape) < 4:
            target = target.unsqueeze(1)
        if len(input.shape) < 4:
            input = input.unsqueeze(1)
        target, input = target.to(device), input.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output = net(input)
        
        # Compute the loss
        loss = criterion(output[0], target)
        # if i == 0:
        #     fig, ax = plt.subplots(1, 2, figsize = (30, 30))
        #     ax[0].imshow(output[0].detach().cpu().numpy()[0, 0, :, :], cmap = 'gray')
        #     ax[1].imshow(target.detach().cpu().numpy()[0, 0, :, :], cmap = 'gray')
        #     plt.show()
        #     plt.savefig('test.png', bbox_inches = 'tight')
        #     plt.close()        
        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(trainloader)}")
    if loss.item() < prev_best:
        prev_best = loss.item()
        os.makedirs('ckpts/unet', exist_ok=True)
        torch.save(net.state_dict(), f'ckpts/unet/best.pth')

print('Training finished.')
