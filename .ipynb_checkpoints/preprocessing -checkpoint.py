import numpy as np
from PIL import Image
import glob
import tqdm
import os
import torch
import cv2
import matplotlib.pyplot as plt
import word2vec
img_size = 64
def img_preprocess(img):
    size=(img_size,img_size)
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
def label_preprocess(text):
    global tag_vectors
    text=text.replace("-"," ")
    tokens=text.split()
    if tokens==None:
        print("Notoken")
    tag_vectors = np.array([catch(token) for token in tokens if catch(token) != None])
    if len(tag_vectors)==0:
        tag_vectors=np.zeros(300).reshape(-1,300)
    tag_vectors=tag_vectors.astype("float32")
    # for idx,token in enumerate(tokens):
    #     try:
    #         tag_vector=Embedding_model[token].reshape(-1,300)
    #     except KeyError:
    #         # print(idx,token)
    #         continue
    #     if idx==0:
    #         tag_vectors=tag_vector
    #     else:
    #         tag_vectors=np.concatenate((tag_vectors,tag_vector),0)
    #         print(tag_vectors.dtype)
    return tag_vectors
def save_np(filepath,ndarray,object="img"):
    if object=="img":
        dir_name="./Myfont/dataset/img_vector/"
        file_name=os.path.splitext(os.path.basename(filepath))[0]
        file_name=file_name[:-3]
    elif object=="tag":
        dir_name="./Myfont/dataset/tag_vector/"
        file_name = os.path.basename(filepath)
    else:
        raise ValueError("オブジェクと指定されてない")
    np.save(dir_name+file_name+".npy",ndarray)
def save_pt(filepath,ndarray,object="img"):
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
paths = './Myfont/dataset/fontimage'  # 各フォルダに同じラベル付けをしたい画像を入れる
img_files = sorted(glob.glob('./Myfont/dataset/fontimage/*_AA.png'))  # Aのフォントのpathを記録
tag_files = sorted(glob.glob('./Myfont/dataset/taglabel/*'))

Embedding_model=word2vec.word2vec()


#os.mkdir("./Myfont/dataset/taglabel/tag_vector")
for cnt_file,(i_f,t_f) in tqdm.tqdm(enumerate(zip(img_files,tag_files)),total=len(img_files)):
    img = np.array(Image.open(i_f).convert("L")) # すべての画像の読み込み
    img = img_preprocess(img)
    img = img.reshape(-1,img_size, img_size)
    save_pt(i_f, img,object="img")
    with open(t_f, 'r', encoding='utf-8') as f:
        text = f.read()
        all_tag_vectors=label_preprocess(text)
        save_pt(t_f,all_tag_vectors,object="tag")
        # if cnt_file == 0:
        #     tag_label = all_tag_vectors
        # else:
        #     tag_label = np.concatenate((tag_label,all_tag_vectors), 0)
    # with open(, 'r', encoding='utf-8') as f:
    #   text = f.read()
    #   idx = text.find(keyword)
    # print(train.shape)  # trainにベクトル化した画像データをスタックしていく

print("save done")