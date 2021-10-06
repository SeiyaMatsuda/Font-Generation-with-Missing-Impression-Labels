import numpy as np
from PIL import Image
import glob
import tqdm
import os
import cv2
from options import get_parser
import word2vec
from utils.mylib import split_list, pickle_dump

def font_triming(img, img_size=64):
    size=(img_size, img_size)
    try:
        h_max = np.where(img != 255)[0].min()
        h_min = np.where(img != 255)[0].max()
        w_max = np.where(img != 255)[1].min()
        w_min = np.where(img != 255)[1].max()
        img = img[h_max:h_min, w_max:w_min]
        h, w = img.shape
        if w > h:
            support = (w-h) // 2
            img = np.pad(img, [(support, w-h-support), (0, 0)], "constant", constant_values=(255,255))
        elif w < h:
            support = (h-w) // 2
            img = np.pad(img, [(0, 0), (support, h-w-support)], "constant", constant_values=(255,255))
        else:
            pass
        img = cv2.resize(img, size)
    except ValueError:
        img = np.full(size, 255)
    return img

def save_image_vector(dataset_path, font_name, ndarray):
    #画像を処理
    dir_name= os.path.join(dataset_path, "images")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.save(os.path.join(dir_name, font_name + ".npy"), ndarray)
if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    sortsecond = lambda a: os.path.splitext(os.path.basename(a))[0]  # 各フォルダに同じラベル付けをしたい画像を入れる
    img_files = sorted(glob.glob(os.path.join(opts.data_path, 'clean_fontimage/general/*.png')), key=sortsecond)  # 画像ファイルのパスを記録
    img_files = list(split_list(img_files, 52))
    tag_files = [os.path.join(os.path.join(opts.data_path, 'taglabel'),
                              os.path.splitext(os.path.basename(img_files[i][0]))[0].replace('_AA', '')) for i in
                 range(len(img_files))]
    tag_files = sorted(tag_files, key=sortsecond)

    Embedding_model = word2vec.word2vec()
    w2v_vocab = {}
    impression_word_list = []
    dataset_path = './dataset'
    for cnt_file, (i_f, t_f) in tqdm.tqdm(enumerate(zip(img_files, tag_files)), total=len(img_files)):
        data = []
        # for ii in i_f:
        #     img = np.array(Image.open(ii).convert("L"))  # すべての画像の読み込み
        #     img = font_triming(img, opts.img_size)
        #     img = img.reshape(-1, opts.img_size, opts.img_size)
        #     data.append(img)
        # data = np.concatenate(data, axis=0)
        font_name = os.path.splitext(os.path.basename(i_f[0]))[0][:-3]
        save_image_vector(dataset_path, font_name, data)
        with open(t_f, 'r', encoding='utf-8') as f:
            text = f.read()
            all_tag = text.replace("-", "_").split()
            impression_word = []
            for tt in all_tag:
                try:
                    w2v_vocab[tt] = Embedding_model[tt]
                    impression_word.append(tt)
                except KeyError:
                    pass
        impression_word_list.append(impression_word)
    pickle_dump(impression_word_list, os.path.join(dataset_path, 'impression_word_list.pickle'))
    pickle_dump(w2v_vocab,  os.path.join(dataset_path, 'w2v_vocab.pickle'))