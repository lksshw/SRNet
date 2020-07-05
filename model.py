# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import os
from skimage.io import imread, imsave
from skimage.transform import resize
import skimage
import numpy as np
import torch
import torchvision.transforms as tf
from torchvision.models import vgg19
from collections import namedtuple
import cfg


def calc_padding(h, w, k, s):
    
    h_pad = (((h-1)*s) + k - h)//2 
    w_pad = (((w-1)*s) + k - w)//2
    
    return (h_pad, w_pad)

def calc_inv_padding(h, w, k, s):
    h_pad = (k-h + ((h-1)*s))//2
    w_pad = (k-w + ((w-1)*s))//2
    
    return (h_pad, w_pad)


class Conv_bn_block(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.conv = torch.nn.Conv2d(*args, **kwargs)
        self.bn = torch.nn.BatchNorm2d(kwargs['out_channels'])
        
    def forward(self, input):
        return torch.nn.functional.leaky_relu(self.bn(self.conv(input)))

class Res_block(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
            
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels//4, kernel_size = 1, stride =1)
        self.conv2 = torch.nn.Conv2d(in_channels//4, in_channels//16, kernel_size = 3, stride = 1, padding = 1)
        
        self.conv3 = torch.nn.Conv2d(in_channels//16, in_channels, kernel_size = 1, stride=1)
        
        self.bn = torch.nn.BatchNorm2d(in_channels)
       
    def forward(self, x):
        
        xin = x
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.add(xin, x)
        x = torch.nn.functional.leaky_relu(self.bn(x))
        
        return x

class encoder_net(torch.nn.Module):
    
    def __init__(self, in_channels, get_feature_map = False):
        super().__init__()
            
        self.cnum = 32
        self.get_feature_map = get_feature_map
        self.C_block_1 = Conv_bn_block(in_channels = in_channels, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self.C_block_2 = Conv_bn_block(in_channels =self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #--------------------------
        
        self.conv_1 = torch.nn.Conv2d(in_channels = self.cnum, out_channels = 2 * self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(64, 128, 3, 2))
        
        self.C_block_3 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self.C_block_4 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #---------------------------
        
        self.conv_2 = torch.nn.Conv2d(in_channels = 2*self.cnum, out_channels = 4* self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(64, 128, 3, 2))
        
        self.C_block_5 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self.C_block_6 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #---------------------------
        
        self.conv_3 = torch.nn.Conv2d(in_channels = 4*self.cnum, out_channels = 8* self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(64, 128, 3, 2))
        
        self.C_block_7 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self.C_block_8 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
                
        x = self.C_block_1(x)
        x = self.C_block_2(x)
        
        x = torch.nn.functional.leaky_relu(self.conv_1(x))
        x = self.C_block_3(x)
        x = self.C_block_4(x)
        
        f1 = x
        
        x = torch.nn.functional.leaky_relu(self.conv_2(x))
        x = self.C_block_5(x)
        x = self.C_block_6(x)
        
        f2 = x
        
        x = torch.nn.functional.leaky_relu(self.conv_3(x))
        x = self.C_block_7(x)
        x = self.C_block_8(x)
        
        
        if self.get_feature_map:
            return x, [f2, f1]
        
        else:
            return x
        
        
class build_res_block(torch.nn.Module):
    
    def __init__(self, in_channels):
        
        super().__init__()
        
        self.res_block1 = Res_block(in_channels)
        self.res_block2 = Res_block(in_channels)
        self.res_block3 = Res_block(in_channels)
        self.res_block4 = Res_block(in_channels)
        
    def forward(self, x):
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        return x
    
    
class decoder_net(torch.nn.Module):
    
    def __init__(self, in_channels, get_feature_map = False):
        super().__init__()
        
        self.cnum = 32
                
        self.get_feature_map = get_feature_map
        
        self.c_block_1 = Conv_bn_block(in_channels = in_channels , out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1) 
        
        self.vonc_0 = torch.nn.Conv2d(in_channels = 16*self.cnum, out_channels = 8*self.cnum, kernel_size = 1, stride = 1)
        
        self.c_block_2 = Conv_bn_block(in_channels = 8*self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1)
        
        #-----------------
        
        self.deconv_1 = torch.nn.ConvTranspose2d(8*self.cnum, 4*self.cnum, kernel_size = 3, stride = 2, padding = calc_inv_padding(64, 128, 3, 2))
        
        self.c_block_3 = Conv_bn_block(in_channels = 4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self.vonc_1 = torch.nn.Conv2d(in_channels = 8*self.cnum, out_channels = 4*self.cnum, kernel_size = 1, stride = 1)
        
        self.c_block_4 = Conv_bn_block(in_channels = 4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #-----------------
        
        self.deconv_2 = torch.nn.ConvTranspose2d(4*self.cnum, 2*self.cnum, kernel_size =3 , stride = 2, padding = calc_inv_padding(64, 128, 3, 2))
        
        self.c_block_5 = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self.vonc_2 = torch.nn.Conv2d(in_channels = 4*self.cnum, out_channels = 2*self.cnum, kernel_size = 1, stride = 1)
        
        self.c_block_6 = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #----------------
        
        self.deconv_3 = torch.nn.ConvTranspose2d(2*self.cnum, self.cnum, kernel_size =3 , stride = 2, padding = calc_inv_padding(64, 128, 3, 2))
        
        self.c_block_7 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self.c_block_8 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        
    def forward(self, x, fuse = None):
        
        
        if fuse and fuse[0] is not None:
            x = torch.cat((x, fuse[0]), dim = 1)
            x = self.vonc_0(x)
            
        x = self.c_block_1(x)
        x = self.c_block_2(x)
        f1 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self.deconv_1(x))
        
        if fuse and fuse[1] is not None:
            x = torch.cat((x, fuse[1]), dim = 1)
            x = self.vonc_1(x)
            
        x = self.c_block_3(x)
        x = self.c_block_4(x)
        f2 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self.deconv_2(x))
        if fuse and fuse[2] is not None:
            x = torch.cat((x, fuse[2]), dim = 1)
            x = self.vonc_2(x)
        
        x = self.c_block_5(x)
        x = self.c_block_6(x)
        f3 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self.deconv_3(x))
        x = self.c_block_7(x)
        x = self.c_block_8(x)
        
        if self.get_feature_map:
            return x, [f1, f2, f3]        
        
        else:
            return x
                                                  
        
class text_conversion_net(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.cnum = 32
        self.enc_net_1 = encoder_net(in_channels)
        self.res_net_1 = build_res_block(8*self.cnum)
        
        self.enc_net_2 = encoder_net(in_channels)
        self.res_net_2 = build_res_block(8*self.cnum)
        
        self.dec_net_1 = decoder_net(16*self.cnum)
        self.conv_1 = torch.nn.Conv2d(self.cnum, 1, kernel_size =3, stride = 1, padding = 1)
        
        self.dec_net_2 = decoder_net(16*self.cnum)
        self.conv_block_1 = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self.conv_2 = torch.nn.Conv2d(2*self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x_t, x_s):
                
        x_t = self.enc_net_1(x_t)
        x_t = self.res_net_1(x_t)
        
        x_s = self.enc_net_2(x_s)
        x_s = self.res_net_2(x_s)
        
        x = torch.cat((x_t, x_s), dim = 1)

        y_sk = self.dec_net_1(x, fuse = None)
        y_sk_out = torch.sigmoid(self.conv_1(y_sk))        
        
        y_t = self.dec_net_2(x, fuse = None)
        
        y_t = torch.cat((y_sk, y_t), dim = 1)
        y_t = self.conv_block_1(y_t)
        y_t_out = torch.tanh(self.conv_2(y_t))
        
        return y_sk_out, y_t_out
                                          
        
class inpainting_net(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.cnum = 32
        self.enc_net_1 = encoder_net(in_channels, get_feature_map = True)
        self.res_net_1 = build_res_block(8*self.cnum)
        
        self.dec_net_1 = decoder_net(8*self.cnum,  get_feature_map = True)
        self.conv_1 = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        
        x, f_encoder = self.enc_net_1(x)
        x = self.res_net_1(x)
        
        x, fs = self.dec_net_1(x, fuse = [None] + f_encoder)
        
        x = torch.tanh(self.conv_1(x))
        
        return x, fs
  
        
class fusion_net(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnum = 32
        
        self.enc_net_1 = encoder_net(in_channels)
        self.res_net_1 = build_res_block(8*self.cnum)
        
        self.dec_net_1 = decoder_net(8*self.cnum)
        
        self.conv_1 = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
         
    def forward(self, x, fuse):
        
        x = self.enc_net_1(x)
        x = self.res_net_1(x)
        x = self.dec_net_1(x, fuse = fuse)
        x = torch.tanh(self.conv_1(x))
        
        return x
           
class Generator(torch.nn.Module):
    
    def __init__(self, in_channels):
        
        super().__init__()
        
        self.cnum = 32
        
        self.text_conv_net = text_conversion_net(in_channels)
        
        self.inpaint_net = inpainting_net(in_channels)
        
        self.fus_net = fusion_net(in_channels)
        
    def forward(self, i_t, i_s):
                
        o_sk, o_t = self.text_conv_net(i_t, i_s)
        
        o_b, fuse = self.inpaint_net(i_s)
        
        o_f = self.fus_net(o_t, fuse)
        
        return o_sk, o_t, o_b, o_f
    
    
class Discriminator(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.cnum = 32
        self.c_1 = torch.nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 2, padding = calc_padding(64, 128, 3, 2))
        self.c_2 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = calc_padding(64, 128, 3, 2))
    
        self.bn_1 = torch.nn.BatchNorm2d(128)
        
        self.c_3 = torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = calc_padding(64, 128, 3, 2))
        
        self.bn_2 = torch.nn.BatchNorm2d(256)
        
        self.c_4 = torch.nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = calc_padding(64, 128, 3, 2))
        
        self.bn_3 = torch.nn.BatchNorm2d(512)
        
        self.c_5 = torch.nn.Conv2d(512, 1,  kernel_size = 3, stride = 1, padding = 1)
        
        self.bn_4 = torch.nn.BatchNorm2d(1)
        
     
    def forward(self, x):
        
        x = torch.nn.functional.leaky_relu(self.c_1(x))
        x = self.c_2(x)
        x = torch.nn.functional.leaky_relu(self.bn_1(x))
        x = self.c_3(x)
        x = torch.nn.functional.leaky_relu(self.bn_2(x))
        x = self.c_4(x)
        x = torch.nn.functional.leaky_relu(self.bn_3(x))
        x = self.c_5(x)
        x = self.bn_4(x)
        
        return x
        
    
class Vgg19(torch.nn.Module):
    def __init__(self):
        
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained = True).features)
        self.features = torch.nn.ModuleList(features).eval()
        
    def forward(self, x):
        
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results
        
            