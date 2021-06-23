import numpy as np
from PIL import Image
import glob
import tqdm
import os
import torch
import cv2
from options import get_parser
import word2vec
from mylib import split_list, label_preprocess, pickle_dump
def font_triming(img, img_size=64):
    size=(img_size, img_size)
    try:
        h_max = np.where(img != 255)[0].min()
        h_min = np.where(img != 255)[0].max()
        w_max = np.where(img != 255)[1].min()
        w_min = np.where(img != 255)[1].max()
        img = img[h_max:h_min, w_max:w_min]
        h, w = img.shape
        if w>h:
            support = (w-h) // 2
            img=np.pad(img,[(support+2,w-h-support+2),(2,2)],"constant",constant_values=(255,255))
        elif w<h:
            support = (h-w) // 2
            img = np.pad(img, [(2, 2), (support+2, h-w-support+2)], "constant" ,constant_values=(255,255))
        else:
            img=np.pad(img,[(2,2),(2,2)],"constant",constant_values=(255,255))
        img=cv2.resize(img, size)
    except ValueError:
        img=np.full(size,255)

    return img
def catch(token):
    try:
        tag_vector=Embedding_model[token]
        return tag_vector.tolist()
    except KeyError:
        return None
def save_np(filepath, ndarray, object="img"):
    if object=="img":
        dir_name="./Myfont/dataset/clean_fontimage/general/vector"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name=os.path.splitext(os.path.basename(filepath[0]))[0]
        file_name=file_name[:-3]
    elif object=="tag":
        dir_name="./Myfont/dataset/tag_vector/"
        file_name = os.path.basename(filepath)
    else:
        raise ValueError("オブジェクと指定されてない")
    np.save(os.path.join(dir_name, file_name+".npy"), ndarray)
def save_pt(filepath, ndarray, object="img"):
    if object=="img":
        dir_name="./Myfont/dataset/img_vector/"
        file_name=os.path.splitext(os.path.basename(filepath))[0]
        file_name=file_name[:-3]
    elif object=="tag":
        dir_name="./Myfont/dataset/tag_vector/"
        file_name = os.path.basename(filepath)
    else:
        raise ValueError("オブジェクと指定されてない")
    tensor=torch.from_numpy(ndarray).clone()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = tensor.to(device)
    torch.save(tensor,dir_name+file_name+".pt")

if __name__=='__main__':
    parser = get_parser()
    opts = parser.parse_args()
    sortsecond = lambda a: os.path.splitext(os.path.basename(a))[0]
    paths = './Myfont/dataset/clean_fontimage/general'  # 各フォルダに同じラベル付けをしたい画像を入れる
    img_files = sorted(glob.glob('./Myfont/dataset/clean_fontimage/general/*.png'), key=sortsecond) #画像ファイルのパスを記録
    img_files = list(split_list(img_files, 52))
    tag_files = [os.path.join('./Myfont/dataset/taglabel', os.path.splitext(os.path.basename(img_files[i][0]))[0].replace('_AA','')) for i in range(len(img_files))]
    tag_files = sorted(tag_files, key = sortsecond)

    Embedding_model = word2vec.word2vec()
    w2v_vocab = {}
    impression_word_list = []

    #os.mkdir("./Myfont/dataset/taglabel/tag_vector")
    for cnt_file,(i_f,t_f) in tqdm.tqdm(enumerate(zip(img_files, tag_files)), total=len(img_files)):
        data=[]
        for ii in i_f:
            img = np.array(Image.open(ii).convert("L")) # すべての画像の読み込み
            img = font_triming(img,opts.img_size)
            img = img.reshape(-1, opts.img_size, opts.img_size)
            data.append(img)
        data=np.concatenate(data, axis=0)
        print(data.shape)
        save_np(i_f, data, object="img")
    #     with open(t_f, 'r', encoding='utf-8') as f:
    #         text = f.read()
    #         all_tag = label_preprocess(text)
    #         impression_word = []
    #         for tt in all_tag:
    #             try:
    #                 w2v_vocab[tt] = Embedding_model[tt]
    #                 impression_word.append(tt)
    #             except KeyError:
    #                 pass
    #         impression_word_list.append(impression_word)
    # pickle_dump(impression_word_list, os.path.join(paths, 'impression_word_list.pickle'))
    # pickle_dump(w2v_vocab,  os.path.join(paths, 'w2v_vocab.pickle'))
    print("save done")