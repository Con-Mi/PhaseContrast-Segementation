from torchvision import models
from torch import nn
import torch


class ConvCEluGrNorm(nn.Module):
    def __init__(self, inp_chnl, out_chnl):
        super(ConvCEluGrNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=inp_chnl, out_channels=out_chnl, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=16, num_channels=out_chnl)
        self.celu = nn.CELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.celu(out)
        return out

class UpsampleLayer(nn.Sequential):
    def __init__(self, in_chnl, mid_chnl, out_chnl, transp=False):
        super(UpsampleLayer, self).__init__()
        if not transp:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvCEluGrNorm(in_chnl, mid_chnl),
                ConvCEluGrNorm(mid_chnl, out_chnl)
            )
        else:
            self.block = nn.Sequential(
                ConvCEluGrNorm(in_chnl, mid_chnl),
                nn.ConvTranspose2d(in_channels=mid_chnl, out_channels=out_chnl, 
                        kernel_size=4, stride=2, padding=1, bias=False),
                nn.CELU(inplace=True)
            )

class DenseSegmModel(nn.Module):
    def __init__(self, input_channels, num_filters=32, num_classes=1, pretrained=False):
        super(DenseSegmModel, self).__init__()
        encoder = models.densenet121(pretrained=pretrained).features
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.CELU(inplace=True),
            encoder[3]
        )                              
        self.layer2 = encoder[4:6]     
        self.layer3 = encoder[6:8]     
        self.layer4 = encoder[8:10]    
        self.layer5 = encoder[10]
        self.gn = nn.GroupNorm(num_groups=16, num_channels=1024)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.center = UpsampleLayer(in_chnl=1024, mid_chnl=num_filters*8, out_chnl=num_filters*8)
        self.dec5 = UpsampleLayer(1024 + num_filters*8, num_filters*8, num_filters*8)
        self.dec4 = UpsampleLayer(512 + num_filters*8, num_filters*8, num_filters*8)
        self.dec3 = UpsampleLayer(256 + num_filters*8, num_filters*8, num_filters*8)
        self.dec2 = UpsampleLayer(128 + num_filters*8, num_filters*2, num_filters*2)
        self.dec1 = UpsampleLayer(64+num_filters*2, num_filters, num_filters)
        self.dec0 = UpsampleLayer(num_filters, num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        conv5 = self.layer5(conv4)
        out = self.pool(self.gn(conv5))

        center = self.center(out)
        
        dec5 = self.pool(self.dec5(torch.cat([center, conv5], 1)))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        dec0 = self.dec0(dec1)
        return self.final(dec0)

def DenseLinkModel(input_channels, pretrained=False, num_classes=1):
    return DenseSegmModel(input_channels=input_channels, pretrained=pretrained, num_classes=num_classes)