# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""

import os
import cv2
import math
import numpy as np
import pygame
from pygame import freetype
import random
import multiprocessing
import queue
import Augmentor

from . import render_text_mask
from . import colorize
from . import skeletonization
from . import render_standard_text
from . import data_cfg
import pickle as cp

class datagen():

    def __init__(self):
        
        freetype.init()
        cur_file_path = os.path.dirname(__file__)
        
        font_dir = os.path.join(cur_file_path, data_cfg.font_dir)
        self.font_list = os.listdir(font_dir)
        self.font_list = [os.path.join(font_dir, font_name) for font_name in self.font_list]
        self.standard_font_path = os.path.join(cur_file_path, data_cfg.standard_font_path)
        
        color_filepath = os.path.join(cur_file_path, data_cfg.color_filepath)
        self.colorsRGB, self.colorsLAB = colorize.get_color_matrix(color_filepath)
        
        text_filepath = os.path.join(cur_file_path, data_cfg.text_filepath)
        self.text_list = open(text_filepath, 'r').readlines()
        self.text_list = [text.strip() for text in self.text_list]
        
        bg_filepath = os.path.join(cur_file_path, data_cfg.bg_filepath)

        with open(bg_filepath, 'rb') as f:
	        self.bg_list = set(cp.load(f))
        self.bg_list = [data_cfg.temp_bg_path+img_path.strip() for img_path in self.bg_list]
        
        self.surf_augmentor = Augmentor.DataPipeline(None)
        self.surf_augmentor.random_distortion(probability = data_cfg.elastic_rate,
            grid_width = data_cfg.elastic_grid_size, grid_height = data_cfg.elastic_grid_size,
            magnitude = data_cfg.elastic_magnitude)
        
        self.bg_augmentor = Augmentor.DataPipeline(None)
        self.bg_augmentor.random_brightness(probability = data_cfg.brightness_rate, 
            min_factor = data_cfg.brightness_min, max_factor = data_cfg.brightness_max)
        self.bg_augmentor.random_color(probability = data_cfg.color_rate, 
            min_factor = data_cfg.color_min, max_factor = data_cfg.color_max)
        self.bg_augmentor.random_contrast(probability = data_cfg.contrast_rate, 
            min_factor = data_cfg.contrast_min, max_factor = data_cfg.contrast_max)

    def gen_srnet_data_with_background(self):
        
        while True:
            # choose font, text and bg
            font = np.random.choice(self.font_list)
            text1, text2 = np.random.choice(self.text_list), np.random.choice(self.text_list)
            
            upper_rand = np.random.rand()
            if upper_rand < data_cfg.capitalize_rate + data_cfg.uppercase_rate:
                text1, text2 = text1.capitalize(), text2.capitalize()
            if upper_rand < data_cfg.uppercase_rate:
                text1, text2 = text1.upper(), text2.upper()
            bg = cv2.imread(random.choice(self.bg_list))

            # init font
            font = freetype.Font(font)
            font.antialiased = True
            font.origin = True

            # choose font style
            font.size = np.random.randint(data_cfg.font_size[0], data_cfg.font_size[1] + 1)
            font.underline = np.random.rand() < data_cfg.underline_rate
            font.strong = np.random.rand() < data_cfg.strong_rate
            font.oblique = np.random.rand() < data_cfg.oblique_rate

            # render text to surf
            param = {
                        'is_curve': np.random.rand() < data_cfg.is_curve_rate,
                        'curve_rate': data_cfg.curve_rate_param[0] * np.random.randn() 
                                      + data_cfg.curve_rate_param[1],
                        'curve_center': np.random.randint(0, len(text1))
                    }
            surf1, bbs1 = render_text_mask.render_text(font, text1, param)
            param['curve_center'] = int(param['curve_center'] / len(text1) * len(text2))
            surf2, bbs2 = render_text_mask.render_text(font, text2, param)

            # get padding
            padding_ud = np.random.randint(data_cfg.padding_ud[0], data_cfg.padding_ud[1] + 1, 2)
            padding_lr = np.random.randint(data_cfg.padding_lr[0], data_cfg.padding_lr[1] + 1, 2)
            padding = np.hstack((padding_ud, padding_lr))

            # perspect the surf
            rotate = data_cfg.rotate_param[0] * np.random.randn() + data_cfg.rotate_param[1]
            zoom = data_cfg.zoom_param[0] * np.random.randn(2) + data_cfg.zoom_param[1]
            shear = data_cfg.shear_param[0] * np.random.randn(2) + data_cfg.shear_param[1]
            perspect = data_cfg.perspect_param[0] * np.random.randn(2) +data_cfg.perspect_param[1]
            surf1 = render_text_mask.perspective(surf1, rotate, zoom, shear, perspect, padding) # w first
            surf2 = render_text_mask.perspective(surf2, rotate, zoom, shear, perspect, padding) # w first

            # choose a background
            surf1_h, surf1_w = surf1.shape[:2]
            surf2_h, surf2_w = surf2.shape[:2]
            surf_h = max(surf1_h, surf2_h)
            surf_w = max(surf1_w, surf2_w)
            surf1 = render_text_mask.center2size(surf1, (surf_h, surf_w))
            surf2 = render_text_mask.center2size(surf2, (surf_h, surf_w))

            bg_h, bg_w = bg.shape[:2]
            if bg_w < surf_w or bg_h < surf_h:
                continue
            x = np.random.randint(0, bg_w - surf_w + 1)
            y = np.random.randint(0, bg_h - surf_h + 1)
            t_b = bg[y:y+surf_h, x:x+surf_w, :]
            
            # augment surf
            surfs = [[surf1, surf2]]
            self.surf_augmentor.augmentor_images = surfs
            surf1, surf2 = self.surf_augmentor.sample(1)[0]
            
            # bg augment
            bgs = [[t_b]]
            self.bg_augmentor.augmentor_images = bgs
            t_b = self.bg_augmentor.sample(1)[0][0]

            # render standard text
            i_t = render_standard_text.make_standard_text(self.standard_font_path, text2, (surf_h, surf_w))

            # get min h of bbs
            min_h1 = np.min(bbs1[:, 3])
            min_h2 = np.min(bbs2[:, 3])
            min_h = min(min_h1, min_h2)

            # get font color
            if np.random.rand() < data_cfg.use_random_color_rate:
                fg_col, bg_col = (np.random.rand(3) * 255.).astype(np.uint8), (np.random.rand(3) * 255.).astype(np.uint8)
            else:
                fg_col, bg_col = colorize.get_font_color(self.colorsRGB, self.colorsLAB, t_b)

            # colorful the surf and conbine foreground and background
            param = {
                        'is_border': np.random.rand() < data_cfg.is_border_rate,
                        'bordar_color': tuple(np.random.randint(0, 256, 3)),
                        'is_shadow': np.random.rand() < data_cfg.is_shadow_rate,
                        'shadow_angle': np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree)
                                        + data_cfg.shadow_angle_param[0] * np.random.randn(),
                        'shadow_shift': data_cfg.shadow_shift_param[0, :] * np.random.randn(3)
                                        + data_cfg.shadow_shift_param[1, :],
                        'shadow_opacity': data_cfg.shadow_opacity_param[0] * np.random.randn()
                                          + data_cfg.shadow_opacity_param[1]
                    }
            _, i_s = colorize.colorize(surf1, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h, param)
            t_t, t_f = colorize.colorize(surf2, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h, param)
            
            # skeletonization
            t_sk = skeletonization.skeletonization(surf2, 127)
            break
   
        return [i_t, i_s, t_sk, t_t, t_b, t_f, surf2]

def enqueue_data(queue, capacity):  
    
    np.random.seed()
    gen = datagen()
    while True:
        try:
            data = gen.gen_srnet_data_with_background()
        except Exception as e:
            pass
        if queue.qsize() < capacity:
            queue.put(data)

class multiprocess_datagen():
    
    def __init__(self, process_num, data_capacity):
        
        self.process_num = process_num
        self.data_capacity = data_capacity
            
    def multiprocess_runningqueue(self):
        
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()
        self.pool = multiprocessing.Pool(processes = self.process_num)
        self.processes = []
        for _ in range(self.process_num):
            p = self.pool.apply_async(enqueue_data, args = (self.queue, self.data_capacity))
            self.processes.append(p)
        self.pool.close()
        
    def dequeue_data(self):
        
        while self.queue.empty():
            pass
        data = self.queue.get()
        return data
        '''
        data = None
        if not self.queue.empty():
            data = self.queue.get()
        return data
        '''

    def dequeue_batch(self, batch_size, data_shape):
        
        while self.queue.qsize() < batch_size:
            pass

        i_t_batch, i_s_batch = [], []
        t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
        mask_t_batch = []
        
        for i in range(batch_size):
            i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = self.dequeue_data()
            i_t_batch.append(i_t)
            i_s_batch.append(i_s)
            t_sk_batch.append(t_sk)
            t_t_batch.append(t_t)
            t_b_batch.append(t_b)
            t_f_batch.append(t_f)
            mask_t_batch.append(mask_t)
        
        w_sum = 0
        for t_b in t_b_batch:
            h, w = t_b.shape[:2]
            scale_ratio = data_shape[0] / h
            w_sum += int(w * scale_ratio)
        
        to_h = data_shape[0]
        to_w = w_sum // batch_size
        to_w = int(round(to_w / 8)) * 8
        to_size = (to_w, to_h) # w first for cv2
        for i in range(batch_size): 
            i_t_batch[i] = cv2.resize(i_t_batch[i], to_size)
            i_s_batch[i] = cv2.resize(i_s_batch[i], to_size)
            t_sk_batch[i] = cv2.resize(t_sk_batch[i], to_size, interpolation=cv2.INTER_NEAREST)
            t_t_batch[i] = cv2.resize(t_t_batch[i], to_size)
            t_b_batch[i] = cv2.resize(t_b_batch[i], to_size)
            t_f_batch[i] = cv2.resize(t_f_batch[i], to_size)
            mask_t_batch[i] = cv2.resize(mask_t_batch[i], to_size, interpolation=cv2.INTER_NEAREST)
            # eliminate the effect of resize on t_sk
            t_sk_batch[i] = skeletonization.skeletonization(mask_t_batch[i], 127)

        i_t_batch = np.stack(i_t_batch)
        i_s_batch = np.stack(i_s_batch)
        t_sk_batch = np.expand_dims(np.stack(t_sk_batch), axis = -1)
        t_t_batch = np.stack(t_t_batch)
        t_b_batch = np.stack(t_b_batch)
        t_f_batch = np.stack(t_f_batch)
        mask_t_batch = np.expand_dims(np.stack(mask_t_batch), axis = -1)
        
        i_t_batch = i_t_batch.astype(np.float32) / 127.5 - 1. 
        i_s_batch = i_s_batch.astype(np.float32) / 127.5 - 1. 
        t_sk_batch = t_sk_batch.astype(np.float32) / 255. 
        t_t_batch = t_t_batch.astype(np.float32) / 127.5 - 1. 
        t_b_batch = t_b_batch.astype(np.float32) / 127.5 - 1. 
        t_f_batch = t_f_batch.astype(np.float32) / 127.5 - 1.
        mask_t_batch = mask_t_batch.astype(np.float32) / 255.
        
        return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch]
    
    def get_queue_size(self):
        
        return self.queue.qsize()
    
    def terminate_pool(self):
        
        self.pool.terminate()
