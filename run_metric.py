import gc
import random
import sys
import os

import pylab as pl

sys.path.append(os.path.join(os.path.dirname("__file__"), "../"))
print(sys.path)
from models.PGmodel import Generator
from models.Imp2Font import Imp2Font
from utils.metrics import FID, GAN_train_test
from utils.font_generation import Font_Generator
import pandas as pd
import word2vec
import matplotlib.pyplot as plt
import string
from PIL import Image
from utils.mylib import *
from dataset import *
import numpy as np
import tqdm
import argparse

parser = argparse.ArgumentParser(description='定量評価のパラメータ設定')
parser.add_argument('--work_dir', help='作業を行うディレクトリ')    # 必須の引数を追加
parser.add_argument('--how_map', default="mAP_imp", help='mAPの計算の方法')    # 必須の引数を追加
parser.add_argument('--model_path', help='生成モデルのファイル名', default="weight_iter_90000.pth")
parser.add_argument('--font_generation', dest='font_generation', action='store_true')
parser.add_argument('--no_font_generation', dest='font_generation', action='store_false')
parser.set_defaults(font_generation=True)
parser.add_argument('--img_save', dest='img_save', action='store_true')
parser.add_argument('--no_img_save', dest='img_save', action='store_false')
parser.set_defaults(img_save=False)
parser.add_argument('--FID', dest='FID', action='store_true')
parser.add_argument('--no_FID', dest='FID', action='store_false')
parser.set_defaults(FID=False)
parser.add_argument('--Intra_FID', dest='Intra_FID', action='store_true')
parser.add_argument('--no_Intra_FID', dest='Intra_FID', action='store_false')
parser.set_defaults(Intra_FID=False)
parser.add_argument('--GAN_train', dest='GAN_train', action='store_true')
parser.add_argument('--no_GAN_train', dest='GAN_train', action='store_false')
parser.set_defaults(GAN_train=False)
parser.add_argument('--GAN_test', dest='GAN_test', action='store_true')
parser.add_argument('--no_GAN_test', dest='GAN_test', action='store_false')
parser.set_defaults(GAN_test=False)
parser.add_argument('--imp2font', dest='imp2font', action='store_true')
parser.add_argument('--no_imp2font', dest='imp2font', action='store_false')
parser.set_defaults(imp2font=False)
parser.add_argument('--intra_fid_min_num', type=int, help="intra_FIDを計算する際の最小頻度", default=200)
parser.add_argument('--reduce_label_ratio', type=float, help="生成の際ラベルを削減する", default=1)
args = parser.parse_args()

class CONFIG:
    work_dir = args.work_dir
    print(work_dir)
    model_path = os.path.join(work_dir, "models", args.model_path)
    FID = args.FID
    Intra_FID = args.Intra_FID
    font_generation = args.font_generation
    GAN_train = args.GAN_train
    GAN_test = args.GAN_test
    how_map = args.how_map
    imp2font = args.imp2font
    intra_fid_min_num = args.intra_fid_min_num
    reduce_label_ratio = args.reduce_label_ratio
    img_save = args.img_save

def init_logger():
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
    log_file = CONFIG.work_dir + "/score.log"
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def Generate_Fonts(FG, data, label, reduce_ratio= 1.0, save_dir=None):
    fake_img = []
    ll = random.sample(list(range(len(label))), 5000)
    for idx in tqdm.tqdm(ll, total=len(ll)):
        l = label[idx]
        l =  random.sample(l, int(len(l) * reduce_ratio))
        ff = FG.generate_from_impression(1, l, string.ascii_uppercase, shuffle=True)
        fake_img.append(ff)
    fake_img = torch.cat(fake_img, axis=0)
    fake_img = fake_img.reshape(-1, 26, fake_img.size(2), fake_img.size(3))
    real_img = data.reshape(-1, 26, data.shape[2], data.shape[3])[torch.tensor(ll)]
    if save_dir:
        os.makedirs(os.path.join(save_dir, "real_img"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "fake_img"), exist_ok=True)
        for i1, ff in tqdm.tqdm(enumerate(fake_img), total=len(fake_img)):
            for i2, fff in enumerate(ff):
                save_path = os.path.join(save_dir, "fake_img", f"{str(i1).zfill(5)}_{string.ascii_uppercase[i2]}.png")
                fff = fff.to('cpu').detach().numpy().copy()
                img_fff = Image.fromarray((fff * 255).astype(np.uint8))
                img_fff.save(save_path)
        for i1, rr in tqdm.tqdm(enumerate(real_img), total=len(real_img)):
            for i2, rrr in enumerate(rr):
                save_path = os.path.join(save_dir, "real_img",
                                         f"{str(i1).zfill(5)}_{string.ascii_uppercase[i2]}.png")
                rrr = rrr.to('cpu').detach().numpy().copy()
                img_rrr = Image.fromarray((rrr * 255).astype(np.uint8))
                img_rrr.save(save_path)
    real_img = real_img.reshape(-1, 1, 64, 64)
    fake_img = fake_img.reshape(-1, 1, 64, 64)
    fake_img[fake_img.isnan() == True] = 1
    return real_img, fake_img, ll

def calc_FID(fake_img, real_img):
    fid = FID()
    score = fid.calculate_fretchet(fake_img.expand(-1, 3, -1, -1), real_img.expand(-1, 3, -1, -1), batch_size=128,
                                   cuda=True)
    return score

def calc_Intra_FID(data, ID, label, FG, min_num):
    index = np.full((len(ID.keys()), len(label)), False)
    mask = [list(map(lambda x: ID[x] - 1, l)) for l in label]
    fid = FID()
    intra_fid = {}
    batch_size = 128
    for i, m in enumerate(mask):
        index[m, i] = True
    for i, key in tqdm.tqdm(enumerate(ID.keys()), total=len(ID.keys())):
        real_img = data[index[i]]
        if min_num > len(real_img):
            continue
        thr = torch.randperm(len(real_img))[:min_num]
        real_img = real_img[thr].view(-1, 1, data.size(2), data.size(3)).expand(-1, 3, -1, -1)
        fake_img = FG.generate_from_impression(min_num, [key], string.ascii_uppercase, shuffle=True)
        fake_img = fake_img.reshape(-1, 1, fake_img.size(2), fake_img.size(3)).expand(-1, 3, -1, -1)
        print(real_img.max(), fake_img.max(), real_img.min(), fake_img.min())
        fid_score = fid.calculate_fretchet(fake_img/2+0.5, real_img/2+0.5, batch_size=batch_size, cuda=True, verbose=False)
        intra_fid[key] = fid_score

        print("FID-score:{} :::::{}".format(key, fid_score))
    intra_fid_df = pd.DataFrame({'word': list(intra_fid.keys()), 'FID': list(intra_fid.values())})
    return intra_fid_df["FID"].mean(), intra_fid_df

def calc_GAN_train(real_img, fake_img, ID,  label, ll, metric="mAP_imp"):
    gtt = GAN_train_test(len(ID), ID, save_dir=CONFIG.work_dir, train_or_test="train", metric=metric)
    pos_weight = gtt.caliculate_pos_weight(label)
    label = [label[i] for i in ll]
    gan_train_score, AP_train_earch_imp, validation_score  = gtt.run(real_img.reshape(-1, 26, 64, 64), label,
                                                  fake_img.reshape(-1, 26, 64, 64), label, pos_weight, 100,
                                                  batch_size=64, shuffle=True)
    return gan_train_score, AP_train_earch_imp, validation_score

def calc_GAN_test(real_img, fake_img, ID, label, ll, metric="mAP_imp"):
    gtt = GAN_train_test(len(ID), ID, save_dir=CONFIG.work_dir, train_or_test="test", metric=metric)
    pos_weight = gtt.caliculate_pos_weight(label)
    label = [label[i] for i in ll]
    gan_test_score, AP_test_earch_imp, validation_score = gtt.run(real_img.reshape(-1, 26, 64, 64), label,
                                                                  fake_img.reshape(-1, 26, 64, 64), label, pos_weight, 100,
                                                                  batch_size=64, shuffle=True)
    return gan_test_score, AP_test_earch_imp, validation_score

def main():
    SEED = 1111
    # SEED = 1000
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger = init_logger()
    parser = get_parser()
    opts = parser.parse_args(args=[])
    data = torch.from_numpy(np.array([np.load(d) for d in opts.data]))[:,:26, :, :].float()/127.5-1
    label = opts.impression_word_list
    if CONFIG.imp2font:
        Embedding_model = word2vec.word2vec()
        weights = Embedding_model.vectors
        ID = {}
        c = 1
        mask = []
        for idx, key in enumerate(Embedding_model.vocab.keys()):
            if key in opts.w2v_vocab.keys():
                ID[key] = c
                c += 1
                mask.append(idx)
            else:
                continue
        weights = weights[mask]
        G_model = Imp2Font(weights, latent_size=300, w2v_dimension=300, imp_num=1574, char_num=26,sum_weight=False, deepsets=False, w2v=False)
        G_model.load_state_dict(fix_model_state_dict(torch.load(CONFIG.model_path, map_location='cuda:0')["G_model_state_dict"]), strict=False)
        G_model.eval()
    else:
        ID = {key: idx + 1 for idx, key in enumerate(opts.w2v_vocab)}
        weights = np.array(list(opts.w2v_vocab.values()))
        G_model = Generator(weights, latent_size=256, w2v_dimension=300, num_dimension=100, char_num=26,
                            attention=False,
                            normalize=True)
        G_model.load_state_dict(fix_model_state_dict(torch.load(CONFIG.model_path, map_location='cuda:0')["G_net"]), strict=False)
    G_model.to('cuda')
    FG = Font_Generator(G_model, ID, device='cuda', imp2font=CONFIG.imp2font)
    if CONFIG.font_generation:
        if CONFIG.img_save:
            save_dir = CONFIG.work_dir
        else:
            save_dir = None
        real_img, fake_img, label_selected = Generate_Fonts(FG, data, label, reduce_ratio=CONFIG.reduce_label_ratio, save_dir=save_dir)
    if CONFIG.FID:
        print("============caliculate FID============")
        fid_score = calc_FID(fake_img/2+0.5, real_img/2+0.5)
        logger.info('FID:{}'.format(fid_score))
    if CONFIG.Intra_FID:
        print("============caliculate Intra FID============")
        intra_fid_score, intra_fid_df = calc_Intra_FID(data, ID, label, FG, min_num=CONFIG.intra_fid_min_num)
        intra_fid_df.to_csv(CONFIG.work_dir +f'/intra_fid_score_{CONFIG.intra_fid_min_num}.csv', index=False)
        logger.info('Intra_FID(min_num{}):{}'.format(CONFIG.intra_fid_min_num, intra_fid_score))
    del FG, G_model
    gc.collect()
    if CONFIG.GAN_train:
        print("============caliculate GAN train============")
        GAN_train_score, AP_train_earch, _ = calc_GAN_train(real_img/2+0.5, fake_img/2+0.5,  ID, label, label_selected, metric=CONFIG.how_map)
        if CONFIG.how_map == "mAP_imp":
            AP_train_earch_imp_df = pd.DataFrame(
                {'word': list(ID.keys()), 'AP': np.mean(AP_train_earch, axis=0).tolist()})
            AP_train_earch_imp_df.to_csv(CONFIG.work_dir +'/AP_train_earch_imp.csv', index=False)
        elif CONFIG.how_map=="mAP_default":
            AP_train_earch_font_df = pd.DataFrame(
                {'fonts': list(np.array(opts.data)[label_selected]), "labels":np.array(label)[label_selected], 'AP':np.mean(AP_train_earch, axis=0).tolist()})
            AP_train_earch_font_df.to_csv(CONFIG.work_dir + '/AP_train_earch_font.csv', index=False)
        logger.info('GAN_train_score({}):{} CV:{}'.format(CONFIG.how_map, sum(GAN_train_score)/len(GAN_train_score), GAN_train_score))
    if CONFIG.GAN_test:
        print("============caliculate GAN test============")
        GAN_test_score, AP_test_earch, _ = calc_GAN_test(real_img/2+0.5, fake_img/2+0.5,  ID, label, label_selected,  metric=CONFIG.how_map)
        if CONFIG.how_map=="mAP_imp":
            AP_test_earch_imp_df = pd.DataFrame(
                {'word': list(ID.keys()), 'AP': np.mean(AP_test_earch, axis=0).tolist()})
            if CONFIG.reduce_label_ratio!=1.0:
                if not os.path.isdir(CONFIG.work_dir + f'/GAN_test_reduce_ratio'):
                    os.makedirs(CONFIG.work_dir + f'/GAN_test_reduce_ratio')
                AP_test_earch_imp_df.to_csv(CONFIG.work_dir + f'/GAN_test_reduce_ratio/AP_test_earch_imp_{CONFIG.reduce_label_ratio}.csv', index=False)
                pd.Series(GAN_test_score).to_csv(CONFIG.work_dir + f'/GAN_test_reduce_ratio/GAN_test_CV_score_{CONFIG.reduce_label_ratio}.csv',
                    index=False)
            else:
                AP_test_earch_imp_df.to_csv(CONFIG.work_dir + '/AP_test_earch_imp.csv', index=False)
        elif CONFIG.how_map=="mAP_default":
            AP_test_earch_font_df = pd.DataFrame(
                {'fonts': list(np.array(opts.data)[label_selected]), "labels": np.array(label)[label_selected], 'AP': np.mean(AP_test_earch, axis=0).tolist()})
            AP_test_earch_font_df.to_csv(CONFIG.work_dir + '/AP_test_earch_font.csv', index=False)
        logger.info('GAN_test_score({}):{} CV:{}'.format(CONFIG.how_map, sum(GAN_test_score)/len(GAN_test_score), GAN_test_score))

if __name__ == '__main__':
    main()
