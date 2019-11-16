# author: Niwhskal
# github : https://github.com/test13234/SRNet

import numpy as np
import os
import torch
import torchvision.transforms
from utils import *
from datagen import srnet_datagen, gen_input_data
import cfg
from skimage.transform import resize
from skimage import io
from model import SRNet
from torchvision import models, transforms, datasets


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def custom_collate(batch):
    
    i_t_batch, i_s_batch = [], []
    t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
    mask_t_batch = []
    
    for item in batch:
        
        t_b= item[4]
        
        w_sum = 0
        h, w = t_b.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        w_sum += int(w * scale_ratio)
        
    to_h = cfg.data_shape[0]
    to_w = w_sum // cfg.batch_size
    to_w = int(round(to_w / 8)) * 8
    to_scale = (to_h, to_w)
    
    for item in batch:
   
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = item


        i_t = resize(i_t, to_scale)
        i_s = resize(i_s, to_scale)
        t_sk = np.expand_dims(resize(t_sk, to_scale), axis = -1) 
        t_t = resize(t_t, to_scale)
        t_b = resize(t_b, to_scale)  
        t_f = resize(t_f, to_scale)
        mask_t = np.expand_dims(resize(mask_t, to_scale), axis = -1)


        i_t = i_t.transpose((2, 0, 1))
        i_s = i_s.transpose((2, 0, 1))
        t_sk = t_sk.transpose((2, 0, 1))
        t_t = t_t.transpose((2, 0, 1))
        t_b = t_b.transpose((2, 0, 1))
        t_f = t_f.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1)) 

        i_t_batch.append(i_t) 
        i_s_batch.append(i_s)
        t_sk_batch.append(t_sk)
        t_t_batch.append(t_t) 
        t_b_batch.append(t_b) 
        t_f_batch.append(t_f)
        mask_t_batch.append(mask_t)

    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_sk_batch = np.stack(t_sk_batch)
    t_t_batch = np.stack(t_t_batch)
    t_b_batch = np.stack(t_b_batch)
    t_f_batch = np.stack(t_f_batch)
    mask_t_batch = np.stack(mask_t_batch)

    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.) 
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 
    t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.) 
    t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.) 
    t_b_batch = torch.from_numpy(t_b_batch.astype(np.float32) / 127.5 - 1.) 
    t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.) 
    mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)    

      
    return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch]

def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)

def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    
    train_name = get_train_name()
    
    print_log('Initializing SRNET', content_color = PrintColor['yellow'])
    model = SRNet(shape = cfg.data_shape, name = train_name)
    print_log('model compiled.', content_color = PrintColor['yellow'])
    
    train_data = datagen_srnet(cfg)
    
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate = custom_collate,  pin_memory = True)
    
    trfms = To_tensor()
    example_data = example_dataset(transform = trfms)
        
    example_loader = DataLoader(dataset = example_data, batch_size = len(example_data), shuffle = False)
    
    print_log('training start.', content_color = PrintColor['yellow'])
        
    G = Generator(in_channels = 3).cuda()
    
    D1 = discriminator(in_channels = 6).cuda()
    
    D2 = discriminator(in_channels = 6).cuda()
        
    vgg_features = Vgg19().cuda()    
        
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D2_solver = torch.optim.Adam(D2.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(G_solver, milestones=[30, 200], gamma=0.5)
    
    d1_scheduler = torch.optim.lr_scheduler.MultiStepLR(D1_solver, milestones=[30, 200], gamma=0.5)
    
    d2_scheduler = torch.optim.lr_scheduler.MultiStepLR(D2_solver, milestones=[30, 200], gamma=0.5)
    
    requires_grad(G, False)

    requires_grad(D1, True)
    requires_grad(D2, True)


    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
        
    
    trainiter = iter(train_data)
    example_iter = iter(example_loader)

    for step in tqdm(range(cfg.max_iter)):
        
        D1.zero_grad()
        D2.zero_grad()
        
        if ((step+1) % save_ckpt_interval == 0):
            
            torch.save(
                {
                    'generator': G.module.state_dict(),
                    'discriminator1': D1.module.state_dict(),
                    'discriminator2': D2.module.state_dict(),
                    'g_optimizer': G_solver.state_dict(),
                    'd1_optimizer': D1_solver.state_dict(),
                    'd2_optimizer': D2_solver.state_dict(),
                },
                f'checkpoint/train_step-{step+1}.model',
            )
                
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = trainiter.next()
        
        inputs = [i_t, i_s]
        labels = [t_sk, t_t, t_b, t_f]
        
        o_sk, o_t, o_b, o_f = G(inputs)
        
        i_db_true = torch.cat((t_b, i_s), dim = 1)
        i_db_pred = torch.cat((o_b, i_s), dim = 1)
        
        i_df_true = torch.cat((t_f, i_t), dim = 1)
        i_df_pred = torch.cat((o_f, i_t), dim = 1)
        
        o_db_true = D1(i_db_true)
        o_db_pred = D1(i_db_pred)
        
        o_df_true = D2(i_df_true)
        o_df_pred = D2(i_df_pred)
        
        i_vgg = torch.cat((t_f, o_f), dim = 0)
        
        out_vgg = vgg_features(i_vgg)
        
        db_loss = build_discriminator_loss(o_db_true,  o_db_pred)
        
        df_loss = build_discriminator_loss(o_df_true, o_df_pred)
                
        out_g = [o_sk, o_t, o_b, o_f, mask_t]
        
        out_d = [o_db_true, o_db_pred, o_df_true, o_df_pred]
        
        g_loss, detail = build_generator_loss(out_g, out_d, out_vgg, labels)
        
        db_loss.backward()
        df_loss.backward()
        
        D1_solver.step()
        
        D2_solver.step()
        
        d1_scheduler.step()
        d2_scheduler.step()
        
        clip_grad(D1)
        clip_grad(D2)
        
        
        if ((step+1) % 5 == 0):
            
            requires_grad(G, True)

            requires_grad(D1, False)
            requires_grad(D2, False)
            
            G.zero_grad()
            
            g_loss.backward()
            
            G_solver.step()
            
            g_scheduler.step()
            
            clip_grad(G)
            
            requires_grad(G, False)

            requires_grad(D1, True)
            requires_grad(D2, True)
            
        if ((step+1) % cfg.write_log_interval == 0):
            
            print('Iter: {}/{} | Gen: {} | D_bg: {} | D_fus: {}'.format(step+1, cfg.max_iter, g_loss.item(), db_loss.item(), df_loss.item()))
            
        if ((step+1) % gen_example_interval == 0):
            
            savedir = os.path.join(cfg.example_result_dir, train_name, 'iter-' + str(step+1).zfill(len(str(cfg.max_iter))))
            
            with torch.no_grad():
                
                inp = example_iter.next()
                
                o_sk, o_t, o_b, o_f = G(inp)
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                o_sk = skimage.img_as_ubyte(o_sk)
                o_t = skimage.img_as_ubyte(o_t + 1)
                o_b = skimage.img_as_ubyte(o_b + 1)
                o_f = skimage.img_as_ubyte(o_f + 1)
                
                                           
                                           
                io.imsave(os.path.join(save_dir, name + 'o_f.png'), o_f)
                
                io.imsave(os.path.join(save_dir, name + 'o_sk.png'), o_sk)
                io.imsave(os.path.join(save_dir, name + 'o_t.png'), o_t)
                io.imsave(os.path.join(save_dir, name + 'o_b.png'), o_b)
                
if __name__ == '__main__':
    main()