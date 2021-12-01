import sys
import os
sys.path.append(os.path.join(os.path.dirname("__file__"), "../../"))
print(sys.path)
from models.PGmodel import Generator
from utils.metrics import FID, GAN_train_test
from utils.font_generation import Font_Generator
import pandas as pd
import string
from logging import getLogger
from dataset import *
import numpy as np
import tqdm

class CONFIG:
    model_path = './models/weight_iter_90000.pth'
    FID = False
    Intra_FID = True
    GAN_train = False
    GAN_test = False

def init_logger():
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
    log_file = "./score.log"
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def Generate_Fonts(FG, data, label):
    fake_img = []
    ll = random.sample(list(range(len(label))), 5000)
    for idx in tqdm.tqdm(ll, total=len(ll)):
        l = label[idx]
        ff = FG.generate_from_impression(1, l, string.ascii_uppercase)
        fake_img.append(ff)
    fake_img = torch.cat(fake_img, axis=0)
    fake_img = fake_img.reshape(-1, 26, fake_img.size(2), fake_img.size(3))
    real_img = data.reshape(-1, 26, data.shape[2], data.shape[3])[torch.tensor(ll)]
    real_img = real_img.reshape(-1, 1, 64, 64)
    fake_img = fake_img.reshape(-1, 1, 64, 64)
    fake_img[fake_img.isnan() == True] = 1
    return real_img, fake_img, ll

def calc_FID(fake_img, real_img):
    fid = FID()
    score = fid.calculate_fretchet(fake_img.expand(-1, 3, -1, -1), real_img.expand(-1, 3, -1, -1), batch_size=128,
                                   cuda=True)
    return score

def calc_Intra_FID(data, ID, label, FG):
    index = np.full((len(ID.keys()), len(label)), False)
    mask = [list(map(lambda x: ID[x] - 1, l)) for l in label]
    fid = FID()
    intra_fid = {}
    batch_size = 256
    for i, m in enumerate(mask):
        index[m, i] = True
    for i, key in tqdm.tqdm(enumerate(ID.keys()), total=len(ID.keys())):
        real_img = data[index[i]]
        if 30 > len(real_img):
            continue
        thr = torch.randperm(len(real_img))[:30]
        real_img = real_img[thr].view(-1, 1, data.size(2), data.size(3)).expand(-1, 3, -1, -1)
        fake_img = FG.generate_from_impression(30, [key], string.ascii_uppercase)
        fake_img = fake_img.reshape(-1, 1, fake_img.size(2), fake_img.size(3)).expand(-1, 3, -1, -1)
        print(real_img.max(), fake_img.max(), real_img.min(), fake_img.min())
        fid_score = fid.calculate_fretchet(fake_img, real_img, batch_size=batch_size, cuda=True, verbose=False)
        intra_fid[key] = fid_score
        print("FID-score:{} :::::{}".format(key, fid_score))
    intra_fid_df = pd.DataFrame({'word': list(intra_fid.keys()), 'FID': list(intra_fid.values())})
    return intra_fid_df["FID"].mean(), intra_fid_df

def calc_GAN_train(real_img, fake_img, ID,  label, ll):
    gtt = GAN_train_test(len(ID), ID, train_or_test="train")
    pos_weight = gtt.caliculate_pos_weight(label)
    label = [label[i] for i in ll]
    gan_train_score, AP_train_earch_imp = gtt.run(real_img.reshape(-1, 26, 64, 64), label,
                                                  fake_img.reshape(-1, 26, 64, 64), label, pos_weight, 100,
                                                  batch_size=64, shuffle=True)
    return gan_train_score, AP_train_earch_imp

def calc_GAN_test(real_img, fake_img, ID, label, ll):
    gtt = GAN_train_test(len(ID), ID, train_or_test="test")
    pos_weight = gtt.caliculate_pos_weight(label)
    label = [label[i] for i in ll]
    gan_test_score, AP_test_earch_imp = gtt.run(real_img.reshape(-1, 26, 64, 64), label, fake_img.reshape(-1, 26, 64, 64), label, pos_weight, 100, batch_size=64, shuffle=True)
    return gan_test_score, AP_test_earch_imp

def main():
    SEED = 1111
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
    ID = {key:idx+1 for idx, key in enumerate(opts.w2v_vocab)}
    weights = np.array(list(opts.w2v_vocab.values()))
    G_model = Generator(weights, latent_size=256, w2v_dimension=300, num_dimension=100, char_num=26, attention=False,
                        normalize=True).to('cuda')
    FG = Font_Generator(G_model, CONFIG.model_path, ID, device='cuda')
    real_img, fake_img, label_selected = Generate_Fonts(FG, data, label)
    if CONFIG.FID:
        print("============caliculate FID============")
        fid_score = calc_FID(fake_img, real_img)
        logger.info('FID:{}'.format(fid_score))
    if CONFIG.Intra_FID:
        print("============caliculate Intra FID============")
        intra_fid_score, intra_fid_df = calc_Intra_FID(data, ID, label, FG)
        intra_fid_df.to_csv('./intra_fid_score.csv', index=False)
        logger.info('Intra_FID:{}'.format(intra_fid_score))
    if CONFIG.GAN_train:
        print("============caliculate GAN train============")
        GAN_train_score, AP_train_earch_imp = calc_GAN_train(real_img, fake_img,  ID, label, label_selected)
        AP_train_earch_imp_df = pd.DataFrame(
            {'word': list(ID.keys()), 'FID': np.mean(AP_train_earch_imp, axis=0).tolist()})
        AP_train_earch_imp_df.to_csv('./AP_train_earch_imp.csv', index=False)
        logger.info('GAN_train_score:{} CV:{}'.format(sum(GAN_train_score)/len(GAN_train_score), GAN_train_score))
    if CONFIG.GAN_test:
        print("============caliculate GAN test============")
        GAN_test_score, AP_test_earch_imp = calc_GAN_test(real_img, fake_img,  ID, label, label_selected)
        AP_test_earch_imp_df = pd.DataFrame(
            {'word': list(ID.keys()), 'FID': np.mean(AP_test_earch_imp, axis=0).tolist()})
        AP_test_earch_imp_df.to_csv('./AP_test_earch_imp.csv', index=False)
        logger.info('GAN_test_score:{} CV:{}'.format(sum(GAN_test_score)/len(GAN_test_score), GAN_test_score))

if __name__ == '__main__':
    main()
