import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
import word2vec
import numpy as np
import tqdm
import glob
import random
from models import Generator
from mylib import label_preprocess
#result_dir="./result/2020-07-29 11:03:37.049248"
#result_dir="./result/2020-08-05 11:29:23.464542"
#result_dir="./result/2020-08-09 05:32:02.720571"
result_dir="./result/2020-09-02 14:14:59.212300"
model_PATH= os.path.join(result_dir,"checkpoint_cWGAN/G_model_3000")
log_dir=os.path.join(result_dir,"Generate_img")
print(torch.cuda.is_available())

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
def Generate_img(device=device,z_dim=100,num_dimension=300,number=5000,log_dir =log_dir,mode=0):
        G_model = Generator(z_dim, num_dimension).to(device)
        G_model = nn.DataParallel(G_model)
        G_model.load_state_dict(torch.load(model_PATH)["model_state_dict"])
        version = ["token-to-font-mode", "make-all-font-mode"]
        Embedding_model = word2vec.word2vec()
        #生成に必要な乱数
        if version[mode]=="token-to-font-mode":
            tag_files = sorted(glob.glob('./Myfont/dataset/taglabel/*'))
            def condition_input(Embedding_model,log_dir):
                token = input("作成する印象語:")
                try:
                    Embedding_model[token]
                    try:
                        os.makedirs(os.path.join(log_dir, token))
                    except FileExistsError:
                        pass
                    
                except KeyError:
                    print("この印象語はword2vecにより埋め込むことはできません\nもう一度入力してください")
                    condition_input(Embedding_model,log_dir)
                return token
            token=condition_input(Embedding_model,log_dir)
            taglabel = {}
            for i, t_f in tqdm.tqdm(enumerate(tag_files)):
                with open(t_f, 'r', encoding='utf-8') as f:
                    text = f.read()
                    Font_name = os.path.splitext(os.path.basename(t_f))[0]
                    tag = label_preprocess(text)
                    taglabel[Font_name] = tag
            keys = [k for k, v in taglabel.items() if token in v]
            print("{}を持つフォントの個数: {}個".format(token, len(keys)))
            if len(keys)>=28:
                target = ["./Myfont/dataset/img_vector/{}.pt".format(f) for f in random.sample(keys, 28)]
                real_img = torch.cat([torch.load(t).view(-1,1,64,64) for t in target], 0).data.cpu()
                real_img=(real_img/255).float()
                save_image(real_img, os.path.join(log_dir,"real_{}.png".format(token)), nrow=7)
            elif len(keys)==0:
                pass
            else:
                target = ["./Myfont/dataset/img_vector/{}.pt".format(f) for f in keys]
                real_img = torch.cat([torch.load(t).view(-1,1,64,64) for t in target], 0).data.cpu()
                real_img=(real_img/255).float()
                save_image(real_img[:28], os.path.join(log_dir,"real_{}.png".format(token)), nrow=7)
            labels = Embedding_model[token].reshape(-1, num_dimension)
            labels = torch.from_numpy(labels).float().to(device)
            G_model.eval()
            labels = labels.expand(number, -1)

            with torch.no_grad():
                noise = torch.normal(mean = 0.5, std = 0.2, size = (number, z_dim)).to(device)
                #Generatorでサンプル生成
                samples = G_model(noise, labels).data.cpu()
                samples = (samples/2)+0.5
                print("文字フォントを{}枚生成中...".format(number))
                for l in tqdm.tqdm(range(number),total=number):
                    save_image(samples[l],os.path.join(log_dir,token, 'Generated_%05d.png' % (l)))
                save_image(samples[:28],os.path.join(log_dir,"fake_{}.png".format(token)),nrow = 7)
                print("生成終了")
        elif version[mode]=="make-all-font-mode":
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


if __name__=="__main__":
    Generate_img(mode=0)
