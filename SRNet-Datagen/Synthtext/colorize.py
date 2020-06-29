"""
Colorizing the text mask.
Change the original code to Python3 support and simplifed the code structure.
Original project: https://github.com/ankush-me/SynthText
Author: Ankush Gupta
Date: 2015
"""
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.ndimage as scim
import scipy.ndimage.interpolation as sii
import os
import os.path as osp
import pickle as cp
from PIL import Image
import random
from . import poisson_reconstruct


class Layer(object):

    def __init__(self, alpha, color):

        # alpha for the whole image:
        assert alpha.ndim == 2
        self.alpha = alpha
        [n,m] = alpha.shape[:2]

        color = np.atleast_1d(np.array(color)).astype(np.uint8)
        # color for the image:
        if color.ndim == 1: # constant color for whole layer
            ncol = color.size
            if ncol == 1 : #grayscale layer
                self.color = color * np.ones((n, m, 3), dtype = np.uint8)
            if ncol == 3 :
                self.color = np.ones((n,m,3), dtype = np.uint8) * color[None,None,:]
        elif color.ndim == 2: # grayscale image
            self.color = np.repeat(color[:,:,None], repeats = 3, axis = 2).copy().astype(np.uint8)
        elif color.ndim == 3: #rgb image
            self.color = color.copy().astype(np.uint8)
        else:
            print (color.shape)
            raise Exception("color datatype not understood")

class FontColor(object):

    def __init__(self, colorsRGB, colorsLAB):
       
        self.colorsRGB = colorsRGB
        self.colorsLAB = colorsLAB
        self.ncol = colorsRGB.shape[0]

    def sample_normal(self, col_mean, col_std):
        
        col_sample = col_mean + col_std * np.random.randn()
        return np.clip(col_sample, 0, 255).astype(np.uint8)

    def sample_from_data(self, bg_mat):
        
        bg_orig = bg_mat.copy()
        bg_mat = cv2.cvtColor(bg_mat, cv2.COLOR_RGB2Lab)
        bg_mat = np.reshape(bg_mat, (np.prod(bg_mat.shape[:2]),3))
        bg_mean = np.mean(bg_mat, axis = 0)

        norms = np.linalg.norm(self.colorsLAB - bg_mean[None,:], axis = 1)
        # choose a random color amongst the top 3 closest matches:
        #nn = np.random.choice(np.argsort(norms)[:3]) 
        nn = np.argmin(norms)

        ## nearest neighbour color:
        data_col = self.colorsRGB[np.mod(nn, self.ncol),:]

        # color
        col1 = self.sample_normal(data_col[:3],data_col[3:6])
        col2 = self.sample_normal(data_col[6:9],data_col[9:12])

        if nn < self.ncol:
            return (col2, col1)
        else:
            # need to swap to make the second color close to the input backgroun color
            return (col1, col2)

    def mean_color(self, arr):
        
        col = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        col = np.reshape(col, (np.prod(col.shape[:2]),3))
        col = np.mean(col, axis = 0).astype(np.uint8)
        return np.squeeze(cv2.cvtColor(col[None,None,:], cv2.COLOR_HSV2RGB))

    def invert(self, rgb):
        
        rgb = 127 + rgb
        return rgb

    def complement(self, rgb_color):
        
        col_hsv = np.squeeze(cv2.cvtColor(rgb_color[None,None,:], cv2.COLOR_RGB2HSV))
        col_hsv[0] = col_hsv[0] + 128 #uint8 mods to 255
        col_comp = np.squeeze(cv2.cvtColor(col_hsv[None,None,:], cv2.COLOR_HSV2RGB))
        return col_comp

    def triangle_color(self, col1, col2):
        
        col1, col2 = np.array(col1), np.array(col2)
        col1 = np.squeeze(cv2.cvtColor(col1[None,None,:], cv2.COLOR_RGB2HSV))
        col2 = np.squeeze(cv2.cvtColor(col2[None,None,:], cv2.COLOR_RGB2HSV))
        h1, h2 = col1[0], col2[0]
        if h2 < h1: h1, h2 = h2, h1 #swap
        dh = h2 - h1
        if dh < 127: dh = 255 - dh
        col1[0] = h1 + dh / 2
        return np.squeeze(cv2.cvtColor(col1[None,None,:],cv2.COLOR_HSV2RGB))

    def change_value(self, col_rgb, v_std=50):
        
        col = np.squeeze(cv2.cvtColor(col_rgb[None,None,:], cv2.COLOR_RGB2HSV))
        x = col[2]
        vs = np.linspace(0,1)
        ps = np.abs(vs - x / 255.0)
        ps /= np.sum(ps)
        v_rand = np.clip(np.random.choice(vs, p = ps) + 0.1 * np.random.randn(), 0, 1)
        col[2] = 255 * v_rand
        return np.squeeze(cv2.cvtColor(col[None,None,:], cv2.COLOR_HSV2RGB))

class Colorize(object):

    def __init__(self):
        pass

    def drop_shadow(self, alpha, theta, shift, size, op=0.80):
        
        if size % 2 == 0:
            size -= 1
            size = max(1, size)
        shadow = cv2.GaussianBlur(alpha, (size,size), 0)
        [dx, dy] = shift * np.array([-np.sin(theta), np.cos(theta)])
        shadow = op * sii.shift(shadow, shift = [dx,dy], mode = 'constant', cval = 0)
        return shadow.astype(np.uint8)

    def border(self, alpha, size, kernel_type = 'RECT'):
        
        kdict = {'RECT':cv2.MORPH_RECT, 'ELLIPSE':cv2.MORPH_ELLIPSE,
                 'CROSS':cv2.MORPH_CROSS}
        kernel = cv2.getStructuringElement(kdict[kernel_type], (size, size))
        border = cv2.dilate(alpha, kernel, iterations = 1) # - alpha
        return border

    def blend(self, cf, cb, mode = 'normal'):
        
        return cf

    def merge_two(self, fore, back, blend_type = None):
        
        a_f = fore.alpha / 255.0
        a_b = back.alpha / 255.0
        c_f = fore.color
        c_b = back.color

        a_r = a_f + a_b - a_f*a_b
        if blend_type != None:
            c_blend = self.blend(c_f, c_b, blend_type)
            c_r = (((1-a_f)*a_b)[:,:,None] * c_b
                    + ((1-a_b)*a_f)[:,:,None] * c_f
                    + (a_f*a_b)[:,:,None] * c_blend)
        else:
            c_r = (((1-a_f)*a_b)[:,:,None] * c_b
                    + a_f[:,:,None]*c_f)

        return Layer((255 * a_r).astype(np.uint8), c_r.astype(np.uint8))

    def merge_down(self, layers, blends = None):
        
        nlayers = len(layers)
        if nlayers > 1:
            [n, m] = layers[0].alpha.shape[:2]
            out_layer = layers[-1]
            for i in range(-2, -nlayers-1, -1):
                blend = None
                if blends is not None:
                    blend = blends[i+1]
                    out_layer = self.merge_two(fore = layers[i], back = out_layer, blend_type = blend)
            return out_layer
        else:
            return layers[0]

    def resize_im(self, im, osize):
        return np.array(Image.fromarray(im).resize(osize[::-1], Image.BICUBIC))

    def color_border(self, col_text, col_bg, bordar_color_type, bordar_color_idx, bordar_color_noise):
        
        choice = np.random.choice(3)

        col_text = cv2.cvtColor(col_text, cv2.COLOR_RGB2HSV)
        col_text = np.reshape(col_text, (np.prod(col_text.shape[:2]), 3))
        col_text = np.mean(col_text,axis = 0).astype(np.uint8)

        vs = np.linspace(0, 1)
        def get_sample(x):
            ps = np.abs(vs - x / 255.0)
            ps /= np.sum(ps)
            v_rand = np.clip(np.random.choice(vs, p = ps) + 0.1 * bordar_color_noise, 0, 1)
            return 255 * v_rand

        # first choose a color, then inc/dec its VALUE:
        if choice == 0:
            # increase/decrease saturation:
            col_text[0] = get_sample(col_text[0]) # saturation
            col_text = np.squeeze(cv2.cvtColor(col_text[None,None,:], cv2.COLOR_HSV2RGB))
        elif choice == 1:
            # get the complementary color to text:
            col_text = np.squeeze(cv2.cvtColor(col_text[None,None,:], cv2.COLOR_HSV2RGB))
            col_text = self.font_color.complement(col_text)
        else:
            # choose a mid-way color:
            col_bg = cv2.cvtColor(col_bg, cv2.COLOR_RGB2HSV)
            col_bg = np.reshape(col_bg, (np.prod(col_bg.shape[:2]), 3))
            col_bg = np.mean(col_bg,axis = 0).astype(np.uint8)
            col_bg = np.squeeze(cv2.cvtColor(col_bg[None,None,:], cv2.COLOR_HSV2RGB))
            col_text = np.squeeze(cv2.cvtColor(col_text[None,None,:], cv2.COLOR_HSV2RGB))
            col_text = self.font_color.triangle_color(col_text, col_bg)

        # now change the VALUE channel:        
        col_text = np.squeeze(cv2.cvtColor(col_text[None,None,:], cv2.COLOR_RGB2HSV))
        col_text[2] = get_sample(col_text[2]) # value
        return np.squeeze(cv2.cvtColor(col_text[None,None,:], cv2.COLOR_HSV2RGB))

    def color_text(self, text_arr, bg_arr):
        
        fg_col,bg_col = self.font_color.sample_from_data(bg_arr)
        return Layer(alpha = text_arr, color = fg_col), fg_col, bg_col

    def color(self, text_arr, bg_arr, fg_col, bg_col, colorsRGB, colorsLAB, min_h, param):

        self.font_color = FontColor(colorsRGB, colorsLAB)        

        l_text = Layer(alpha = text_arr, color = fg_col)
        bg_col = np.mean(np.mean(bg_arr, axis = 0), axis = 0)
        l_bg = Layer(alpha = 255 * np.ones_like(text_arr, dtype = np.uint8), color = bg_col)
    
        layers = [l_text]
        blends = []

        # add border:
        if param['is_border']:
            if min_h <= 15 : bsz = 1
            elif 15 < min_h < 30: bsz = 3
            else: bsz = 5
            
            border_a = self.border(l_text.alpha, size = bsz)
            l_border = Layer(border_a, color = param['bordar_color'])
            layers.append(l_border)
            blends.append('normal')

        # add shadow:
        if param['is_shadow']:
            # shadow gaussian size:
            if min_h <= 15 : bsz = 1
            elif 15 < min_h < 30: bsz = 3
            else: bsz = 5

            # shadow angle:
            theta = param['shadow_angle']

            # shadow shift:
            if min_h <= 15 : shift = param['shadow_shift'][0]
            elif 15 < min_h < 30: shift = param['shadow_shift'][1]
            else: shift = param['shadow_shift'][2]

            # opacity:
            op = param['shadow_opacity']

            shadow = self.drop_shadow(l_text.alpha, theta, shift, 3 * bsz, op)
            l_shadow = Layer(shadow, 0)
            layers.append(l_shadow)
            blends.append('normal')

        gray_layers = layers.copy()
        gray_blends = blends.copy()
        l_bg_gray = Layer(alpha=255*np.ones_like(text_arr, dtype = np.uint8), color = (127, 127, 127))
        gray_layers.append(l_bg_gray)
        gray_blends.append('normal')
        l_normal_gray = self.merge_down(gray_layers, gray_blends)

        l_bg = Layer(alpha = 255 * np.ones_like(text_arr, dtype = np.uint8), color = bg_col)
        layers.append(l_bg)
        blends.append('normal')
        l_normal = self.merge_down(layers, blends)

        # now do poisson image editing:
        l_bg = Layer(alpha = 255 * np.ones_like(text_arr, dtype = np.uint8), color = bg_arr)

        # image blit
        l_out = poisson_reconstruct.poisson_blit_images(l_normal.color.copy(), l_bg.color.copy())

        return l_normal_gray.color, l_out

def get_color_matrix(col_file):

    with open(col_file,'rb') as f:
        colorsRGB = cp.load(f, encoding ='latin1')
    ncol = colorsRGB.shape[0]
    colorsLAB = np.r_[colorsRGB[:,0:3], colorsRGB[:,6:9]].astype(np.uint8)
    colorsLAB = np.squeeze(cv2.cvtColor(colorsLAB[None,:,:], cv2.COLOR_RGB2Lab))
    return colorsRGB, colorsLAB

def get_font_color(colorsRGB, colorsLAB, bg_arr):

    font_color = FontColor(colorsRGB, colorsLAB)        
    return font_color.sample_from_data(bg_arr)

def colorize(surf, bg, fg_col, bg_col, colorsRGB, colorsLAB, min_h, param):

    c = Colorize()
    return c.color(surf, bg, fg_col, bg_col, colorsRGB, colorsLAB, min_h, param)
