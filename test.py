# -*- coding: utf-8 -*-
import argparse
import os
import random
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import torch
from utils.font_generation import Font_Generator
import string
import glob
from models.PGmodel import Generator
from utils.mylib import *
def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset option
    parser.add_argument('--label', type=str,default="happy", nargs='*')
    parser.add_argument('--char', type=str, default=string.ascii_letters[26:52])
    parser.add_argument('--g_path', type=str, default="./output/weight/weight_iter_90000.pth")
    parser.add_argument('--g_num', type=int, default=4)
    string.ascii_letters
    parser.add_argument('--out', type=str, default="./output")
    parser.add_argument('--SEED', type=int, default=1111)
    parser.add_argument('--data_path', type=str, default='Myfont/preprocessed_dataset',
                        help='Path of the directory where the original data is stored')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    os.makedirs(opts.out, exist_ok=True)
    sortsecond = lambda a: os.path.splitext(os.path.basename(a))[0]
    opts.data = sorted(glob.glob(os.path.join(opts.data_path, 'images', '*.npy')), key=sortsecond)
    opts.impression_word_list=pickle_load(os.path.join(opts.data_path, 'impression_word_list.pickle'))
    opts.w2v_vocab = pickle_load(os.path.join(opts.data_path, 'w2v_vocab.pickle'))
    imp_num = len(opts.w2v_vocab)
    char_num = len(opts.char)
    ID = {key: idx + 1 for idx, key in enumerate(opts.w2v_vocab)}
    weights = np.array(list(opts.w2v_vocab.values()))
    # 再現性確保のためseed値固定
    SEED = opts.SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    G_model = Generator(weights, latent_size=256, w2v_dimension=300, num_dimension=100, char_num=26, normalize=True).to(
        'cuda')
    G_model.load_state_dict(fix_model_state_dict(torch.load(opts.g_path)["G_net"]), strict=False)
    FG = Font_Generator(G_model, ID, device='cuda', imp2font=False)
    samples = FG.generate_from_impression(opts.g_num, opts.label, opts.char, shuffle=False)
    samples = samples.reshape(-1, 1, samples.size(2), samples.size(3))
    samples = F.interpolate(samples, (128, 128), mode='nearest')
    samples = samples / 2 + 0.5
    file_name = "{}_SEED{}.png".format('_'.join(opts.label), opts.SEED)
    save_image(samples, os.path.join(opts.out, file_name), nrow=char_num, padding=0)