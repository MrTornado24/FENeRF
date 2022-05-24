import math
import torch
import torch.nn as nn
import sys
import curriculums
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean([2, 3])

class AdapterBlock(nn.Module):
    def __init__(self, output_channels, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, input):
        return self.model(input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class StridedResidualConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=2, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)

        identity = self.proj(input)

        y = (y + identity)/math.sqrt(2)
        return y

class StridedDiscriminator(nn.Module):
    def __init__(self, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
        [
            StridedResidualConvBlock(32, 64), # 6 256x256 -> 128x128
            StridedResidualConvBlock(64, 128), # 5 128x128 -> 64x64
            StridedResidualConvBlock(128, 256), # 4 64x64 -> 32x32
            StridedResidualConvBlock(256, 400), # 3 32x32 -> 16x16
            StridedResidualConvBlock(400, 400), # 2 16x16 -> 8x8
            StridedResidualConvBlock(400, 400), # 1 8x8 -> 4x4
            StridedResidualConvBlock(400, 400), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}

        self.pose_layer = nn.Linear(2, 400)


    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x, None, None


class ResidualCCBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)

        identity = self.proj(input)

        y = (y + identity)/math.sqrt(2)
        return y

class CCSDiscriminator(nn.Module):
    def __init__(self, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
        [
            ResidualCCBlock(32, 64), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}

        self.pose_layer = nn.Linear(2, 400)


    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x, None, None


class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
    
    
class ResidualCCBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)

        identity = self.proj(input)

        y = (y + identity)/math.sqrt(2)
        return y
    
class CCSDiscriminator(nn.Module):
    def __init__(self, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
        [
            ResidualCCBlock(32, 64), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        
        self.pose_layer = nn.Linear(2, 400)


    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x, None, None
    
    
    
class CCSEncoderDiscriminator(nn.Module):
    def __init__(self, latent_dim=256, input_dim=3, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
        [
            ResidualCCBlock(32, 64), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(output_channels=32, input_channels=self.input_dim),
            AdapterBlock(output_channels=64, input_channels=self.input_dim),
            AdapterBlock(output_channels=128, input_channels=self.input_dim),
            AdapterBlock(output_channels=256, input_channels=self.input_dim),
            AdapterBlock(output_channels=400, input_channels=self.input_dim),
            AdapterBlock(output_channels=400, input_channels=self.input_dim),
            AdapterBlock(output_channels=400, input_channels=self.input_dim),
            AdapterBlock(output_channels=400, input_channels=self.input_dim)
        ])
        self.final_layer = nn.Conv2d(400, 1 + self.latent_dim + 2, 2)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)
    
        prediction = x[..., 0:1]
        latent = x[..., 1:self.latent_dim + 1]
        position = x[..., self.latent_dim + 1 : self.latent_dim + 3]

        return prediction, latent, position


class CCSDoubleEncoderDiscriminator(nn.Module):
    def __init__(self, latent_geo_dim=256, latent_app_dim=256, input_dim=3, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.latent_geo_dim = latent_geo_dim
        self.latent_app_dim = latent_app_dim
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
        [
            ResidualCCBlock(32, 64), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(output_channels=32, input_channels=self.input_dim),
            AdapterBlock(output_channels=64, input_channels=self.input_dim),
            AdapterBlock(output_channels=128, input_channels=self.input_dim),
            AdapterBlock(output_channels=256, input_channels=self.input_dim),
            AdapterBlock(output_channels=400, input_channels=self.input_dim),
            AdapterBlock(output_channels=400, input_channels=self.input_dim),
            AdapterBlock(output_channels=400, input_channels=self.input_dim),
            AdapterBlock(output_channels=400, input_channels=self.input_dim)
        ])
        self.final_layer = nn.Conv2d(400, 1 + self.latent_geo_dim + self.latent_app_dim + 2, 2)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        
        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']
        
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)
    
        prediction = x[..., 0:1]
        latent_geo = x[..., 1:self.latent_geo_dim + 1]
        latent_app = x[..., self.latent_geo_dim + 1: self.latent_geo_dim + self.latent_app_dim + 1]
        position = x[..., self.latent_geo_dim + self.latent_app_dim + 1: self.latent_geo_dim + self.latent_app_dim + 3]

        return prediction, latent_geo, latent_app, position


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc=22, ndf=64, n_layers=3, norm_layer_opt='spectralinstance',
                 use_sigmoid=False, num_D=3, getIntermFeat=True):
        super(MultiscaleDiscriminator, self).__init__()
        self.epoch = 0
        self.step = 0
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer_opt, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input, **kwargs):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=19, ndf=64, n_layers=3, norm_layer_opt='spectralinstance', use_sigmoid=False,
                 getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        norm_layer = get_norm_layer(norm_layer_opt)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]

        else:
            return self.model(input)


def get_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer