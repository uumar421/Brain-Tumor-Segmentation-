import torch
import torch.nn as nn

from torch.nn import init

def normal_init(m):
    if type(m) == nn.Conv3d:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.GroupNorm:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.Linear:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Module:
        normal_init(m)

def conv3d_block(in_channels, out_channels):
    conv = nn.Sequential(
        nn.GroupNorm(num_groups=4, num_channels=in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding = 1, bias=True),
        nn.GroupNorm(num_groups=4, num_channels=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding = 1, bias=True)
    )
    return conv

def conv3d_strd(in_channels, out_channels, strides):
    return nn.Sequential(nn.Conv3d(in_channels,out_channels, kernel_size = 3, stride = strides, padding = 1, bias=True))

def out_conv3d(in_channels, out_channels):
    return nn.Sequential(nn.Conv3d(in_channels,out_channels, kernel_size = 1, bias=True),
                         nn.Sigmoid()
                        )

def tri_upsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
        nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    )

class unet_3d_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        #down_sampling
        self.InitConv = conv3d_strd(4,32,1)
        self.Dropout = nn.Dropout3d(p=0.2)
        self.EncoderBlock0 = conv3d_block(32,32)
        self.EncoderDown1 = conv3d_strd(32,64,2)
        self.EncoderBlock1 = conv3d_block(64,64)
        self.EncoderDown2 = conv3d_strd(64,128,2)
        self.EncoderBlock2 = conv3d_block(128,128)
        self.EncoderDown3 = conv3d_strd(128,256,2)
        self.EncoderBlock3 = conv3d_block(256,256)
        
        #up_sampling
        self.DecoderUp2 = tri_upsample(256,128)
        self.DecoderBlock2 = conv3d_block(256,128)
        self.DecoderUp1 = tri_upsample(128,64)
        self.DecoderBlock1 = conv3d_block(128,64)
        self.DecoderUp0 = tri_upsample(64,32)
        self.DecoderBlock0 = conv3d_block(64,32)
        
        #output
        self.out = out_conv3d(32,3)
        
    def forward(self,inputs):
        #down_sampling
        x = self.InitConv(inputs)
        x = self.Dropout(x)
        #32
        #1st encoder
        x0 = self.EncoderBlock0(x)
        x0 += x
        #value to be concatenated
        
        x1 = self.EncoderDown1(x0)
        #64
        #2nd encoder
        x2 = self.EncoderBlock1(x1)
        x2 += x1
        x3 = self.EncoderBlock1(x2)
        x3 += x2
        #value to be concatenated
        
        x4 = self.EncoderDown2(x3)
        #128
        #3rd encoder
        x5 = self.EncoderBlock2(x4)
        x5 += x4
        x6 = self.EncoderBlock2(x5)
        x6 += x5
        #value to be concatenated

        x7 = self.EncoderDown3(x6)
        #256
        x8 = self.EncoderBlock3(x7)
        x8 += x7
        x9 = self.EncoderBlock3(x8)
        x9 += x8
        x10 = self.EncoderBlock3(x9)
        x10 += x8
        x11 = self.EncoderBlock3(x10)
        x11 += x10
        
        #up_sample
        #1st upsample --> from 256 to 128
        x11 = self.DecoderUp2(x11)
        x10 = torch.cat([x11,x6],1)
        #256
        
        x10 = self.DecoderBlock2(x10)
        x10 += x11
        
        #2nd upsample --> from 128 to 64
        x10 = self.DecoderUp1(x10)
        x9 = torch.cat([x10,x3],1)
        
        x9 = self.DecoderBlock1(x9)
        x9 += x10
        #64
        #3rd upsample --> from 64 to 32
        x9 = self.DecoderUp0(x9)
        x8 = torch.cat([x9,x0],1)
        
        x8 = self.DecoderBlock0(x8)
        x8 += x9
        
        out = self.out(x8)
        return out
