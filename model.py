# author: Niwhskal
# github : https://github.com/test13234/SRNet

import os
from skimage.io import imread, imsave
from skimage.transform import resize
import skimage
import numpy as np
import torch
import torchvision.transforms as tf
from torchvision.models import vgg16
from collections import namedtuple
import cfg
from loss import discriminator_loss, generator_loss


class Conv_bn_block(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.conv = torch.nn.Conv2d(*args, **kwargs)
        self.bn = torch.nn.BatchNorm2d()
        
    def forward(self, input):
        return torch.nn.functional.leaky_relu(self.bn(self.conv(input)))

class Res_block(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
            
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels//4, kernel_size = 1, stride =1)
        self.conv2 = torch.nn.Conv2d(in_channels//4, in_channels//16, kernel_size = 3, stride = 1)
        
        self.conv3 = torch.nn.Conv2d(in_channels//16, in_channels, kernel_size = 1, stride=1)
        
        self.bn = torch.nn.BatchNorm2d()
       
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
            
        self.get_feature_map = get_feature_map
        self.C_block_1 = Conv_bn_block(in_channels, out_channels = self.cnum, kernel_size = 3, stride = 1)
        
        self.C_block_2 = Conv_bn_block(in_channels =self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1)
        
        #--------------------------
        
        self.conv_1 = torch.nn.Conv2d(in_channels = self.cnum, out_channels = 2 * self.cnum, kernel_size = 3, stride = 2)
        
        self.C_block_3 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1)
        
        self.C_block_4 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1)
        
        #---------------------------
        
        self.conv_2 = torch.nn.Conv2d(in_channels = 2*self.cnum, out_channels = 4* self.cnum, kernel_size = 3, stride = 2)
        
        self.C_block_5 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1)
        
        self.C_block_6 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1)
        
        #---------------------------
        
        self.conv_3 = torch.nn.Conv2d(in_channels = 4*self.cnum, out_channels = 8* self.cnum, kernel_size = 3, stride = 2)
        
        self.C_block_7 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1)
        
        self.C_block_8 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1)
        
    def forward(self, x):
        
        x = self.C_block_1(x)
        x = self.C_block_2(x)
        
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = self.C_block_3(x)
        x = self.C_block_4(x)
        
        f1 = x
        
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.C_block_5(x)
        x = self.C_block_6(x)
        
        f2 = x
        
        x = torch.nn.functional.leaky_relu(self.conv3(x))
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
                
        self.get_feature_map = get_feature_map
        
        self.c_block_1 = Conv_bn_block(in_channels, 8*self.cnum, kernel_size = 3, stride =1) 
        
        self.c_block_2 = Conv_bn_block(8*self.cnum, 8*self.cnum, kernel_size = 3, stride =1)
        
        #-----------------
        
        self.deconv_1 = torch.nn.ConvTranspose2d(8*self.cnum, 4*self.cnum, kernel_size - 3, stride = 2)
        
        self.c_block_3 = Conv_bn_block(4*self.cnum, 4*self.cnum, kernel_size = 3, stride = 1)
        self.c_block_4 = Conv_bn_block(4*self.cnum, 4*self.cnum, kernel_size = 3, stride = 1)
        
        #-----------------
        
        self.deconv_2 = torch.nn.ConvTranspose2d(4*self.cnum, 2*self.cnum, kernel_size =3 , stride = 2)
        
        self.c_block_5 = Conv_bn_block(2*self.cnum, 2*self.cnum, kernel_size = 3, stride = 1)
        self.c_block_6 = Conv_bn_block(2*self.cnum, 2*self.cnum, kernel_size = 3, stride = 1)
        
        #----------------
        
        self.deconv_3 = torch.nn.ConvTranspose2d(2*self.cnum, self.cnum, kernel_size =3 , stride = 2)
        
        self.c_block_7 = Conv_bn_block(self.cnum, self.cnum, kernel_size = 3, stride = 1)
        self.c_block_8 = Conv_bn_block(self.cnum, self.cnum, kernel_size = 3, stride = 1)
        
        
    def forward(self, x, fuse = None):
        
        if fuse and fuse[0] is not None:
            x = torch.cat((x, fuse[0]), dim = 1)
        
        x = self.c_block_1(x)
        x = self.c_block_2(x)
        f1 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self.deconv_1(x))
        if fuse and fuse[1] is not None:
            x = torch.cat((x, fuse[1]), dim = 1)
            
        x = self.c_block_3(x)
        x = self.c_block_4(x)
        f2 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self.deconv_2(x))
        if fuse and fuse[2] is not None:
            x = torch.cat((x, fuse[2]), dim = 1)
        
        x = self.c_block_5(x)
        x = self.c_block_6(x)
        f3 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self.deconv_3(x))
        x = self.c_block_7(x)
        x = self.c_block_8(x)
        
        if get_feature_map:
            return x, [f1, f2, f3]        
        
        else:
            return x
                                                  
        
class text_conversion_net(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.enc_net_1 = encoder_net(in_channels)
        self.res_net_1 = build_res_block(8*self.cnum)
        
        self.enc_net_2 = encoder_net(in_channels)
        self.res_net_2 = build_res_block(8*self.cnum)
        
        self.dec_net_1 = decoder_net(16*self.cnum)
        self.conv_1 = torch.nn.Conv2d(self.cnum, 1, kernel_size =3, stride = 1)
        
        self.dec_net_2 = decoder_net(1)
        self.conv_block_1 = Conv_bn_block(2*self.cnum, 2*self.cnum, kernel_size = 3, stride = 1)
        
        self.conv_2 = torch.nn.Conv2d(2*self.cnum, 3, kernel_size = 3, stride = 1)
        
    
    def forward(self, x_t, x_s):
        
        x_t = self.enc_net_1(x_t)
        x_t = self.res_net_1(x_t)
        
        x_s = self.enc_net_2(x_s)
        x_s = self.res_net_2(x_s)
        
        x = torch.cat((x_t, x_s), dim = 1)
        
        y_sk = self.dec_net_1(x, fuse = None)
        y_sk_out = torch.nn.functional.sigmoid(self.conv_1(y_sk))
        
        y_t = self.dec_net_2(x, fuse = None)
        y_t = torch.cat((y_sk, y_t), dim = 1)
        y_t = self.conv_block_1(y_t)
        y_t_out = torch.nn.functional.tanh(self.conv_2(y_t))
        
        return y_sk_out, y_t_out
                                          
        
class inpainting_net(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.enc_net_1 = encoder_net(in_channels, get_feature_map = True)
        self.res_net_1 = build_res_block(8*self.cnum)
        
        self.dec_net_1 = decoder_net(8*self.cnum,  get_feature_map = True)
        self.conv_1 = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1)
        
    def forward(self, x):
        
        x, f_encoder = self.enc_net_1(x)
        x = self.res_net_1(x)
        
        x, fs = self.dec_net_1(x, fuse = [None] + f_encoder)
        
        x = torch.nn.functional.tanh(self.conv_1(x))
        
        return x, fs
  

        
        
class fusion_net(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.enc_net_1 = encoder_net(in_channels)
        self.res_net_1 = build_res_block(8*self.cnum)
        
        self.dec_net_1 = decoder_net(8*self.cnum)
        
        self.conv_1 = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1)
         
    def forward(self, x, fuse):
        
        x = self.enc_net_1(x)
        x = self.res_net_1(x)
        x = self.dec_net_1(x, fuse = fuse)
        x = torch.nn.functional.tanh(self.conv_1(x))
        
        return x
           
class Generator(torch.nn.Module):
    
    def __init__(self, in_channels):
        
        super().__init__()
        
        self.text_conv_net = text_conversion_net(in_channels)
        
        self.inpaint_net = inpainting_net(in_channels)
        
        self.fus_net = fusion_net(in_channels)
        
    def forward(self, i_t, i_s):
        
        o_sk, o_t = self.text_conv_net(i_t, i_s)
        
        o_b, fuse = self.inpaint_net(i_s)
        
        o_f = fus_net(o_t, fuse)
        
        return o_sk, o_t, o_b, o_f
    
    
class Discriminator():
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.c_1 = torch.nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 2)
        self.c_2 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 2)
    
        self.bn_1 = torch.nn.BatchNorm2d()
        
        self.c_3 = torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 2)
        
        self.bn_2 = torch.nn.BatchNorm2d()
        
        self.c_4 = torch.nn.Conv2d(256, 512, kernel_size = 3, stride = 2)
        
        self.bn_3 = torch.nn.BatchNorm2d()
        
        self.c_5 = torch.nn.Conv2d(512, 1,  kernel_size = 3, stride = 1)
        
        self.bn_4 = torch.nn.BatchNorm2d()
        
     
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
        
        
class SRNet(torch.nn.Module):
    
    def __init__(self, shape = [128, 128], in_channels = 3, disc_in_channels = 6):
        super().__init__()
        
        self.name = name
        self.cnum = 32
        self.gen = self.Generator(in_channels)
        self.desc1 = self.Descriminator(disc_in_channels)
        self.desc2 = self.Descriminator(disc_in_channels)
        
        
    def forward(self, i_t, i_s, t_sk, t_t, t_b, t_f, mask_t):
        
        labels = [t_sk, t_t, t_b, t_f]
        
        o_sk, o_t, o_b, o_f = self.gen(i_t, i_s)
        
        i_db_true = torch.cat((t_b, i_s), dim = 1)
        i_db_pred = torch.cat((o_b, i_s), dim = 1)
        #i_db = torch.cat((i_db_true, i_db_pred), dim = 0)
        
        i_df_true = torch.cat((t_f, i_t), dim = 1)
        i_df_pred = torch.cat((o_f, i_t), dim = 1)
        #i_df = torch.cat((i_df_true, i_df_pred), dim = 0)
        
        o_db_true = self.desc1(i_db_true)
        o_db_pred = self.desc1(i_db_pred)
        
        o_df_true = self.desc2(i_df_true)
        o_df_pred = self.desc2(i_df_pred)
        
        gen_op = [o_sk, o_t, o_b, o_f]
        
        desc_op = [o_db_true, o_db_pred, o_df_true, o_df_pred]
        
        return  gen_op, desc_op, labels
    
class Vgg19(torch.nn.Module):
    def __init__(self):
        
        super(vgg19, self).__init__()
        features = list(vgg19(pretrained = True).features)
        self.features = nn.ModuleList(features).eval()
        
    def forward(self, x):
        
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results
        
            