# Predict script 
# author: Niwhskal

import os
import argparse
import cfg
import torch
from tqdm import tqdm
import numpy as np
from model import Generator, Discriminator, Vgg19
from utils import *
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = cfg.example_data_dir)
    parser.add_argument('--save_dir', help = 'Directory to save result', default = cfg.predict_result_dir)
    parser.add_argument('--checkpoint', help = 'ckpt', default = cfg.ckpt_path)
    args = parser.parse_args()

    assert args.input_dir is not None
    assert args.save_dir is not None
    assert args.checkpoint is not None

    print_log('model compiling start.', content_color = PrintColor['yellow'])

    G = Generator(in_channels = 3).to(device)
    D1 = Discriminator(in_channels = 6).to(device)
    D2 = Discriminator(in_channels = 6).to(device)  
    vgg_features = Vgg19().to(device)   
      
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D1_solver = torch.optim.Adam(D1.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D2_solver = torch.optim.Adam(D2.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))

    checkpoint = torch.load(args.checkpoint)
    G.load_state_dict(checkpoint['generator'])
    D1.load_state_dict(checkpoint['discriminator1'])
    D2.load_state_dict(checkpoint['discriminator2'])
    G_solver.load_state_dict(checkpoint['g_optimizer'])
    D1_solver.load_state_dict(checkpoint['d1_optimizer'])
    D2_solver.load_state_dict(checkpoint['d2_optimizer'])

    trfms = To_tensor()
    example_data = example_dataset(data_dir= args.input_dir, transform = trfms)
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)
    example_iter = iter(example_loader)

    print_log('Model compiled.', content_color = PrintColor['yellow'])

    print_log('Predicting', content_color = PrintColor['yellow'])

    G.eval()
    D1.eval()
    D2.eval()

    with torch.no_grad():

      for step in tqdm(range(len(example_data))):

        try:

          inp = example_iter.next()

        except StopIteration:

          example_iter = iter(example_loader)
          inp = example_iter.next()

        i_t = inp[0].to(device)
        i_s = inp[1].to(device)
        name = str(inp[2][0])

        o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

        o_sk = o_sk.squeeze(0).detach().to('cpu')
        o_t = o_t.squeeze(0).detach().to('cpu')
        o_b = o_b.squeeze(0).detach().to('cpu')
        o_f = o_f.squeeze(0).detach().to('cpu')

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        o_sk = F.to_pil_image(o_sk)
        o_t = F.to_pil_image((o_t + 1)/2)
        o_b = F.to_pil_image((o_b + 1)/2)
        o_f = F.to_pil_image((o_f + 1)/2)
                        
        o_f.save(os.path.join(args.save_dir, name + 'o_f.png'))

        #Uncomment the following if you need to save the rest of the predictions

        #o_sk.save(os.path.join(args.save_dir, name + 'o_sk.png'))
        #o_t.save(os.path.join(savedir, name + 'o_t.png'))
        #o_b.save(os.path.join(savedir, name + 'o_b.png'))

            
if __name__ == '__main__':
    main()
    print_log('predicting finished.', content_color = PrintColor['yellow'])

