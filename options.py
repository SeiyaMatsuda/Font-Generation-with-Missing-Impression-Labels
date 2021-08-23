import argparse
import glob
import torch
import os
import datetime
from mylib import pickle_load
def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset option
    parser.add_argument('--mode', type=str, choices=["C", "AC", "CP", 'Imp2Font', "PG"],default='PG')
    parser.add_argument('--emb', type=str, choices=["w2v", "one-hot"], default='w2v')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--num_dimension', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--res_step', type=int, default=5000)
    parser.add_argument('--char_num', type=int, default=26)
    parser.add_argument('--device_count', type=int, default=torch.cuda.device_count())
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--d_lr', type=float, default=0.0002)
    parser.add_argument('--c_lr', type=float, default=0.0005)
    parser.add_argument('--D_num_critic', type=int, default=1)
    parser.add_argument('--G_num_critic', type=int, default=1)
    parser.add_argument('--lambda_gp', type=int, default=5)
    parser.add_argument('--num_iterations', type=int, default=100000)
    parser.add_argument('--num_iterations_decay', type=int, default=100000)
    parser.add_argument('--dt_now', type =str ,default=str(datetime.datetime.now()))
    cuda = True if torch.cuda.is_available() else False
    parser.add_argument('--device', type=str, default=torch.device("cuda" if cuda else "cpu"))
    parser.add_argument('--Tensor', default=torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    parser.add_argument('--LongTensor', default=torch.cuda.LongTensor if cuda else torch.LongTensor)
    parser.add_argument("--lambda_l1", type=float, default=50.0, help='pixel l1 loss lambda')
    parser.add_argument("--lambda_char", type=float, default=5.0, help='char class loss lambda')
    parser.add_argument("--lambda_GAN", type=float, default=5.0, help='GAN loss lambda')
    parser.add_argument("--lambda_cx", type=float, default=5.0, help='Contextual loss lambda')
    parser.add_argument("--lambda_imp", type=float, default=20.0, help='discriminator predict attribute loss lambda')
    parser.add_argument('--root', type=str, default='./result', help='directory contrains the data and outputs')
    parser.add_argument('--out_res', type=int, default=64, help='The resolution of final output image')
    parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
    path = os.path.join(os.path.dirname(__file__), '../', 'dataset')
    print(path)
    if os.path.isdir(path):
        sortsecond = lambda a: os.path.splitext(os.path.basename(a))[0]
        data = sorted(glob.glob(os.path.join(path, 'images', '*.npy')),key = sortsecond)
        parser.add_argument('--data', default = data)
        parser.add_argument('--impression_word_list', type=list, default=pickle_load(os.path.join(path, 'impression_word_list.pickle')))
        parser.add_argument('--correct_impression_word_list', type=list, default=pickle_load(os.path.join(path, 'correct_impression_word_list.pickle')))
        w2v_vocab = pickle_load(os.path.join(path, 'w2v_vocab.pickle'))
        parser.add_argument('--w2v_vocab', type=dict, default=w2v_vocab)
        parser.add_argument('--num_impression_word', type=int, default=len(w2v_vocab))
    return parser

