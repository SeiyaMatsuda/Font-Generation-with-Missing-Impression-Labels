import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
import word2vec
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import glob
from options import get_parser
import random
from dataset import *
from models .DCmodel import ACGenerator
from PIL import Image
from mylib import label_preprocess, Multilabel_OneHot, tile

parser = get_parser()
opts = parser.parse_args()
data = np.array([np.load(d) for d in opts.data])
result_dir= './result/2021-01-19 06:53:32.089775'
model_PATH= os.path.join(result_dir,"checkpoint_cpGAN/model_30")
log_dir=os.path.join(result_dir,"Generate_img")
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
try:
    os.makedirs(log_dir)
except FileExistsError:
    pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def condition_input(ID):
    token = input("作成する印象語:")
    try:
        ID[token]
    except KeyError:
        print("この印象語はword2vecにより埋め込むことはできません\nもう一度入力してください")
        token = condition_input(ID)
    return token
def Generate_img(device=device, z_dim=opts.z_dim, num_dimension=300, char_num = 26, num_generate=30, log_dir =log_dir, mode='generate'):
        G_model = ACGenerator(weights, mask, z_dim=z_dim, char_num=char_num).to(device)
        G_model = nn.DataParallel(G_model)
        G_model.load_state_dict(torch.load(model_PATH)["G_model_state_dict"])
        #生成に必要な乱数
        if mode=="generate":
            train_data = Myfont_dataset(data, opts.impression_word_list, ID, char_num=26, transform = Transform)
            dataloader = torch.utils.data.DataLoader(train_data, batch_size=opts.char_num, shuffle = False,
                                                     collate_fn=collate_fn, drop_last=True)
            tmp = dataloader.__iter__()
            tokens = []
            end =  ''
            print('印象語を入力する場合，xを入力\n入力を終了する場合,qを入力')
            while end != 'q':
                tokens.append(condition_input(ID))
                end = input('印象語を入力する場合，xを入力\n入力を終了する場合,qを入力')
            target = [torch.from_numpy(data[0]).clone() for data in train_data.dataset if not set(tokens).isdisjoint(set(data[1]))]
            print("{}を持つフォントの個数: {}個".format(','.join(tokens), len(target)))
            if len(target) > 0:
                real_img = torch.stack(target)
                save_image(real_img, os.path.join(log_dir,"real_{}.png".format('_'.join(tokens))), nrow=7)
            else:
                pass
            labels = Multilabel_OneHot([[ID[token] for token in tokens]], len(ID), normalize=True)
            G_model.eval()
            labels = labels.expand(num_generate * char_num, -1).to(device)
            char_class = torch.eye(char_num).repeat(num_generate, 1).to(device)
            with torch.no_grad():
                noise = torch.normal(mean = 0.5, std = 0.2, size = (num_generate, z_dim))
                noise = tile(noise, 0, opts.char_num).to(device)
                #Generatorでサンプル生成
                samples = G_model(noise, labels, char_class)
                samples = samples.data.cpu()
                samples = (samples/2)+0.5
                print("文字フォントを{}枚生成中...".format(num_generate))
                save_image(samples,os.path.join(log_dir,"fake_{}.png".format('_'.join(tokens))),nrow = char_num)
                print("生成終了")
        elif mode=="make-all-font-mode":
            with torch.no_grad():
                noise = torch.normal(mean=0.5, std=0.2, size=(1, z_dim)).to(device)
                tag_files = sorted(glob.glob("./Myfont/dataset/tag_vector/*.pt"))
                try:
                    os.makedirs(os.path.join(log_dir,"All_Font"))
                except FileExistsError:
                    pass
                for file in tqdm.tqdm(tag_files,total=len(tag_files)):
                    G_model.eval()

                    with torch.no_grad():
                        label=torch.mean(torch.load(file).float(),axis=0).view(-1,num_dimension).to(device)
                        Font_name=os.path.splitext(os.path.basename(file))[0]
                        sample=G_model(noise,label).data.cpu()
                        sample=(sample / 2) + 0.5
                        save_image(sample, os.path.join(log_dir,"All_Font",'fake_{}.png'.format(Font_name)))
        elif mode == "visualize":
            with torch.no_grad():
                G_model.eval()
                noise = torch.normal(mean=0.5, std=0.2, size=(1, z_dim)).to(device)
                condition1=condition_input(Embedding_model,log_dir)
                condition2=condition_input(Embedding_model, log_dir)
                condition1 = Embedding_model[condition1].reshape(-1, num_dimension)
                condition2 = Embedding_model[condition2].reshape(-1, num_dimension )
                alpha_list=np.linspace(0,1,300)
                samples = []
                for alpha in alpha_list:
                    condition = alpha * condition1 + (1-alpha) * condition2
                    condition = torch.tensor(condition).to(device)
                    sample = G_model(noise, condition)
                    sample = sample.reshape(64, 64).cpu().detach().numpy()
                    samples.append(Image.fromarray(((sample/2)+0.5)*255.0))
                samples[0].save(os.path.join(log_dir, 'randomwalk.gif'), save_all=True, append_images=samples, duration=10, loop=0)

if __name__=="__main__":
    Generate_img(char_num=opts.char_num, mode='generate')
