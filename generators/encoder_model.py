   
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import EqualLinear, ConvLayer, ResBlock
import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, input_dim, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.feature_dim = 512

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, self.feature_dim, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, self.feature_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, self.feature_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, self.feature_dim, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
                conv3x3(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )

class ToStyleCode(nn.Module):
    def __init__(self, n_convs, input_dim=512, out_dim=512):
        super(ToStyleCode, self).__init__()
        self.convs = nn.ModuleList()
        self.out_dim = out_dim

        for i in range(n_convs):
            if i == 0:
                self.convs.append(
                nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=2))
                #self.convs.append(nn.BatchNorm2d(out_dim))
                #self.convs.append(nn.InstanceNorm2d(out_dim))
                self.convs.append(nn.LeakyReLU(inplace=True))
            else:
                self.convs.append(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=2))
                self.convs.append(nn.LeakyReLU(inplace=True))
        
        self.convs = nn.Sequential(*self.convs)
        self.linear = EqualLinear(out_dim, out_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_dim)
        x = self.linear(x)
        return x


class ToStyleHead(nn.Module):
    def __init__(self, input_dim=512, out_dim=512):
        super(ToStyleHead, self).__init__()
        self.out_dim = out_dim

        self.convs = nn.Sequential(
            conv3x3_bn_relu(input_dim, input_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            # output 1x1
            nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0],self.out_dim)
        return x

class FPNEncoder(nn.Module):
    def __init__(self, input_dim, n_latent=14, use_style_head=False, style_layers=[4,5,6]):
        super(FPNEncoder, self).__init__()

        self.n_latent = n_latent
        num_blocks = [3,4,6,3] #resnet 50
        self.FPN_module = FPN(input_dim, Bottleneck, num_blocks)
        # course block 0-2, 4x4->8x8
        self.course_styles = nn.ModuleList()
        for i in range(3):
            if use_style_head:
                self.course_styles.append(ToStyleHead())
            else:
                self.course_styles.append(ToStyleCode(n_convs=style_layers[0]))
        # medium1 block 3-6 16x16->32x32
        self.medium_styles = nn.ModuleList()
        for i in range(4):
            if use_style_head:
                self.medium_styles.append(ToStyleHead())
            else:
                self.medium_styles.append(ToStyleCode(n_convs=style_layers[1]))
        # fine block 7-13 64x64->256x256
        self.fine_styles = nn.ModuleList()
        for i in range(n_latent - 7):
            if use_style_head:
                self.fine_styles.append(ToStyleHead())
            else:
                self.fine_styles.append(ToStyleCode(n_convs=style_layers[2]))

    def forward(self, x):
        styles = []
        # FPN feature
        p2, p3, p4 = self.FPN_module(x)
        
        for style_map in self.course_styles:
            styles.append(style_map(p4))

        for style_map in self.medium_styles:
            styles.append(style_map(p3))
            
        for style_map in self.fine_styles:
            styles.append(style_map(p2))

        styles = torch.stack(styles, dim=1)

        return styles


class ResEncoder(nn.Module):
    def __init__(self, size, input_dim, output_dim, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * channel_multiplier,
                128: 128 * channel_multiplier,
                256: 64 * channel_multiplier,
                512: 32 * channel_multiplier,
                1024: 16 * channel_multiplier,
        }


        convs = [ConvLayer(input_dim, self.channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = self.channels[size]

        for i in range(log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        # self.n_latent = n_latent
        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, self.channels[4], 3)
        # self.final_linear = EqualLinear(self.channels[4] * 4 * 4, n_latent * 512)
        # debug
        self.final_linear = EqualLinear(output_dim)

    def _cal_stddev(self, x):
        batch, channel, height, width = x.shape
        group = min(batch, self.stddev_group)
        stddev = x.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        x = torch.cat([x, stddev], 1)

        return x

    def forward(self, input):
        batch = input.shape[0]
        
        out = self.convs(input)
        
        out = self._cal_stddev(out)

        out = self.final_conv(out)

        # out = out.view(batch, -1)
        # debug
        n_channel = out.shape[1]
        out = out.permute(0,2,3,1).view(-1, n_channel)
        out = self.final_linear(out)

        # out = out.view(batch, self.n_latent, -1)
        frequencies = out[..., :out.shape[-1]//2]
        phase_shifts = out[..., out.shape[-1]//2:]

        return frequencies, phase_shifts


def main():
      real_img = torch.randn(1,3,128,128).cuda()
      model = ResEncoder(size=128, input_dim=3, n_latent=14, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]).cuda()
      out = model(real_img)
      print(out.shape)


if __name__ == '__main__':
      main()
