#author: Niwhskal
#https://github.com/Niwhskal


import torch
import cfg


def build_discriminator_loss(x_true, x_fake):

    d_loss = -torch.mean(torch.log(torch.clamp(x_true, cfg.epsilon, 1.0)) + torch.log(torch.clamp(1.0 - x_fake, cfg.epsilon, 1.0)))
    return d_loss

def build_dice_loss(x_t, x_o):
       
    iflat = x_o.view(-1)
    tflat = x_t.view(-1)
    intersection = (iflat*tflat).sum()
    
    return 1. - torch.mean((2. * intersection + cfg.epsilon)/(iflat.sum() +tflat.sum()+ cfg.epsilon))

def build_l1_loss(x_t, x_o):
        
    return torch.mean(torch.abs(x_t - x_o))

def build_l1_loss_with_mask(x_t, x_o, mask):
    
    mask_ratio = 1. - mask.view(-1).sum() / torch.size(mask)
    l1 = torch.abs(x_t - x_o)
    return mask_ratio * torch.mean(l1 * mask) + (1. - mask_ratio) * torch.mean(l1 * (1. - mask))

def build_perceptual_loss(x):        
    l = []
    for i, f in enumerate(x):
        l.append(build_l1_loss(f[0], f[1]))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_gram_matrix(x):

    x_shape = x.shape
    c, h, w = x_shape[1], x_shape[2], x_shape[3]
    matrix = x.view((-1, c, h * w))
    matrix1 = torch.transpose(matrix, 1, 2)
    gram = torch.matmul(matrix, matrix1) / (h * w * c)
    return gram

def build_style_loss(x):
        
    l = []
    for i, f in enumerate(x):
        f_shape = f[0].shape[0] * f[0].shape[1] *f[0].shape[2]
        f_norm = 1. / f_shape
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        l.append(f_norm * (build_l1_loss(gram_true, gram_pred)))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_vgg_loss(x):
        
    splited = []
    for i, f in enumerate(x):
        splited.append(torch.split(f, 2))
    l_per = build_perceptual_loss(splited)
    l_style = build_style_loss(splited)
    return l_per, l_style

def build_gan_loss(x_pred):
    
    gen_loss = -torch.mean(torch.log(torch.clamp(x_pred, cfg.epsilon, 1.0)))
    
    return gen_loss

def build_generator_loss(out_g, out_d, out_vgg, labels):
        
    o_sk, o_t, o_b, o_f, mask_t = out_g
    o_db_pred, o_df_pred = out_d
    o_vgg = out_vgg
    t_sk, t_t, t_b, t_f = labels
    
    #skeleton loss

    l_t_sk = cfg.lt_alpha * build_dice_loss(t_sk, o_sk)
    l_t_l1 = build_l1_loss(t_t, o_t)
    l_t =  l_t_l1 + l_t_sk
    
    #Background Inpainting module loss

    l_b_gan = build_gan_loss(o_db_pred)
    l_b_l1 = cfg.lb_beta * build_l1_loss(t_b, o_b)
    l_b = l_b_gan + l_b_l1
    
    l_f_gan = build_gan_loss(o_df_pred)
    l_f_l1 = cfg.lf_theta_1 * build_l1_loss(t_f, o_f)
    l_f_vgg_per, l_f_vgg_style = build_vgg_loss(o_vgg)
    l_f_vgg_per = cfg.lf_theta_2 * l_f_vgg_per
    l_f_vgg_style = cfg.lf_theta_3 * l_f_vgg_style
    l_f = l_f_gan + l_f_vgg_per + l_f_vgg_style + l_f_l1
    
    l = cfg.lt * l_t + cfg.lb * l_b + cfg.lf * l_f
    return l, [l_t_sk, l_t_l1, l_b_gan, l_b_l1, l_f_gan, l_f_l1, l_f_vgg_per, l_f_vgg_style]