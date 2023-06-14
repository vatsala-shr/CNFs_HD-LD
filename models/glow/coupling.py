import torch
import torch.nn as nn
import torch.nn.functional as F
from models.glow.act_norm import ActNorm


class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.
    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, cond_channels, mid_channels, 2 * in_channels)
        # self.nn = UNet(inp_channels=(in_channels + cond_channels), 
        #                op_channels=2*in_channels)
        # self.nn = UNet(inp_channels=in_channels,
        #              cond_channels=cond_channels, 
        #              op_channels=2*in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, x_cond, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id, x_cond)
        # st = self.nn(torch.concat([x_id, x_cond], dim = 1))
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj
    
class CondCoupling(nn.Module):
    def __init__(self, in_channels, cond_channels):
        super(CondCoupling, self).__init__()
        self.nn = UNet1(cond_channels, 2 * in_channels)
        # self.nn = NN(inp_channels=(in_channels + cond_channels), 
        #                op_channels=2*in_channels)
        # self.nn = NN(in_channels=cond_channels,
        #              cond_channels=cond_channels,
        #              mid_channels=64,  
        #              out_channels=2*in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, x_cond, ldj, reverse=False):
        # st = self.nn(x_cond, x_cond)
        st = self.nn(x_cond)
        # st = self.nn(torch.concat([x_id, x_cond], dim = 1))
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)
        x_change = x

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        return x_change, ldj

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        x = self.conv3(x)
        x = self.bn3(x) 
        out = out + x
        out = self.relu(out)
        return out

class NN1(nn.Module):
    def __init__(self, inp_channels = 4, cond_channels = 4, op_channels = 1, features = [32, 64, 128]):
        super(NN1, self).__init__()
        self.unet1 = UNet(cond_channels, cond_channels, features = [32, 64, 128], weights = True)
        self.unet2 = UNet((inp_channels + cond_channels), op_channels, features = features)

    def forward(self, x, x_cond):
        attn_weights = self.unet1(x_cond)
        x = torch.concat([x, attn_weights], dim = 1)
        return self.unet2(x)

class DualResidualBlock(nn.Module):
    def __init__(self, inp_channels, cond_channels, op_channels):
        super(DualResidualBlock, self).__init__()
        self.rb1 = ResidualBlock(inp_channels, op_channels)
        self.rb2 = ResidualBlock(cond_channels, op_channels)
    def forward(self, x, x_cond):
        x = self.rb1(x)
        addn = self.rb2(x_cond)
        return x + addn

class UNet1(nn.Module):
    def __init__(self, inp_channels = 2, op_channels = 1, features = [8, 16, 32, 64, 128]):
        super(UNet1, self).__init__()
        
        # Encoder
        self.encoder1 = ResidualBlock(inp_channels, features[0])
        self.encoder2 = ResidualBlock(features[0], features[1])
        self.encoder3 = ResidualBlock(features[1], features[2])
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(features[2], features[1])
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(features[1], features[0])        
        
        # Output
        self.output = nn.Conv2d(features[0], op_channels, kernel_size=1, bias = True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)   

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        
        # Decoder
        dec2 = self.upconv2(enc3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output
        output = self.output(dec1)
        
        return output


class UNet(nn.Module):
    def __init__(self, inp_channels = 2, cond_channels = 2, op_channels = 1, features = [32, 64, 128]):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = DualResidualBlock(inp_channels=inp_channels, 
                                          cond_channels=cond_channels, 
                                          op_channels=features[0])
        self.encoder2 = DualResidualBlock(features[0], 4 * cond_channels, features[1])
        self.encoder3 = DualResidualBlock(features[1], 16 * cond_channels, features[2])
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = DualResidualBlock(features[2], 4 * cond_channels, features[1])
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = DualResidualBlock(features[1], cond_channels, features[0])        
        
        # Output
        self.output = nn.Conv2d(features[0], op_channels, kernel_size=1, bias = True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)    

    def forward(self, x, x_cond):
        # Encoder
        enc1 = self.encoder1(x, x_cond)
        enc2 = self.encoder2(self.pool(enc1), squeeze(x_cond))
        enc3 = self.encoder3(self.pool(enc2), squeeze(squeeze(x_cond)))
        
        # Decoder
        dec2 = self.upconv2(enc3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2, squeeze(x_cond))
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1, x_cond)
        
        # Output
        output = self.output(dec1)

        # if self.weights:
        output = F.sigmoid(output)
        
        return output

class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, cond_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in_condconv = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)
        nn.init.normal_(self.in_condconv.weight, 0., 0.05)

        self.mid_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.mid_condconv1 = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv1.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.mid_condconv2 = nn.Conv2d(cond_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.elu = nn.ELU()

    def forward(self, x, x_cond):
        x = self.in_norm(x)
        x = self.in_conv(x) + self.in_condconv(x_cond)
        x = self.elu(x)

        x = self.mid_conv1(x) + self.mid_condconv1(x_cond)
        x = self.mid_norm(x)
        x = self.elu(x)

        x = self.mid_conv2(x) + self.mid_condconv2(x_cond)
        x = self.out_norm(x)
        x = self.elu(x)

        x = self.out_conv(x)

        return x

def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x