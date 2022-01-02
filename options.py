import argparse
import glob
import torch
import os
import datetime
from utils.mylib import pickle_load
def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset option
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--w2v_dimension', type=int, default=300)
    parser.add_argument('--num_dimension', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--start_iterations', type=int, default=0)
    parser.add_argument('--res_step', type=int, default=15000)
    parser.add_argument('--char_num', type=int, default=26)
    parser.add_argument('--device_count', type=int, default=torch.cuda.device_count())
    parser.add_argument('--g_lr', type=float, default=0.0005)
    parser.add_argument('--d_lr', type=float, default=0.0005)
    parser.add_argument('--multi_learning', dest='multi_learning', action='store_true')
    parser.add_argument('--no_multi_learning', dest='multi_learning', action='store_false')
    parser.set_defaults(multi_learning=True)
    parser.add_argument('--label_transform', dest='label_transform', action='store_true')
    parser.add_argument('--no_label_transform', dest='label_transform', action='store_false')
    parser.set_defaults(label_transform=True)
    parser.add_argument('--label_compress', dest='label_compress', action='store_true')
    parser.add_argument('--no_label_compress', dest='label_compress', action='store_false')
    parser.set_defaults(label_compress=True)
    parser.add_argument('--style_discriminator', dest='style_discriminator', action='store_true')
    parser.add_argument('--no_style_discriminator', dest='style_discriminator', action='store_false')
    parser.add_argument('--reduce_ratio', type=float, default=0.7)
    parser.set_defaults(style_discriminator=True)
    parser.add_argument('--sc_normalize', dest='sc_normalize', action='store_true')
    parser.add_argument('--no_sc_normalize', dest='sc_normalize', action='store_false')
    parser.set_defaults(sc_normalize=True)
    parser.add_argument('--visualize_sc', type=bool, default=False)
    parser.add_argument('--num_critic', type=int, default=1)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_class', type=float, default=10)
    parser.add_argument('--lambda_drift', type=float, default=0.001)
    parser.add_argument('--lambda_style', type=float, default=0.01)
    parser.add_argument('--num_iterations', type=int, default=100000)
    parser.add_argument('--num_iterations_decay', type=int, default=100000)
    parser.add_argument('--min_impression_num', type=int, default=0)
    parser.add_argument('--dt_now', type=str ,default=str(datetime.datetime.now()))
    parser.add_argument('--nibuchan', type=str, default=False)
    parser.add_argument('--label_list', type=list, default=["decorative", "big", "shade", "manuscript", "ghost"])
    cuda = True if torch.cuda.is_available() else False
    parser.add_argument('--gpu_id', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--device', type=str, default=torch.device(f"cuda:{parser.gpu_id[0]}" if cuda else "cpu"))
    parser.add_argument('--Tensor', default=torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    parser.add_argument('--LongTensor', default=torch.cuda.LongTensor if cuda else torch.LongTensor)
    parser.add_argument('--root', type=str, default='./result', help='directory contrains the data and outputs')
    parser.add_argument('--out_res', type=int, default=64, help='The resolution of final output image')
    parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
    parser.add_argument('--data_path', type=str, default='../Myfont/dataset', help='Path of the directory where the original data is stored')
    path = os.path.join(os.path.dirname(__file__), 'dataset')
    if os.path.isdir(path):
        sortsecond = lambda a: os.path.splitext(os.path.basename(a))[0]
        data = sorted(glob.glob(os.path.join(path, 'images', '*.npy')), key=sortsecond)
        parser.add_argument('--data', default=data)
        parser.add_argument('--impression_word_list', type=list, default=pickle_load(os.path.join(path, 'impression_word_list.pickle')))
        w2v_vocab = pickle_load(os.path.join(path, 'w2v_vocab.pickle'))
        parser.add_argument('--w2v_vocab', type=dict, default=w2v_vocab)
        parser.add_argument('--num_impression_word', type=int, default=len(w2v_vocab))
    return parser

