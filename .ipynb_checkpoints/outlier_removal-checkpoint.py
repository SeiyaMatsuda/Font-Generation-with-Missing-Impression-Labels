import numpy as np
import glob
from sklearn.neighbors import LocalOutlierFactor
from mylib import pseudo_hamming, pickle_load, pickle_dump
import os
import tqdm
import random
def main(data, y, w2v_vocab):
    data = data[:, 0, :, :]
    neigh =  LocalOutlierFactor(
                                n_neighbors=5,
                                metric=pseudo_hamming,
                               novelty=True,
                               contamination='auto')
    for token in tqdm.tqdm(sorted(w2v_vocab.keys(),reverse = True), total = len(w2v_vocab.keys())):
        conditional_index = [idx for idx, yy in enumerate(y) if token in yy]
        conditional_data = np.stack([data[idx] for idx in conditional_index])
        conditional_data_flatt = (conditional_data.reshape(conditional_data.shape[0],-1))
        if len(conditional_data_flatt) >= 50:
            a = random.sample(list(range(len(conditional_data_flatt))), 50)
            f = np.array([0] * len(conditional_data_flatt), dtype=bool)
            f[a] = True
            train = conditional_data_flatt[f]
        else:
            train = conditional_data_flatt
        neigh.fit(train)
        d = neigh.predict(conditional_data_flatt)
        index = np.where(d == -1)[0]
        if index.size == 0:
            continue
        else:
            for i in index:
                y[conditional_index[i]].remove(token)
    return y

if __name__ == '__main__':
    paths = './Myfont/dataset/clean_fontimage/general'
    sortsecond=lambda a : os.path.splitext(os.path.basename(a))[0]
    img_files = sorted(glob.glob(os.path.join(paths, 'vector', '*.npy')), key=sortsecond)
    data=np.stack([np.load(img_file) for img_file in img_files])
    y = pickle_load(os.path.join(paths, 'impression_word_list.pickle'))
    w2v_vocab = pickle_load(os.path.join(paths, 'w2v_vocab.pickle'))
    new_y = main(data, y, w2v_vocab)
    pickle_dump(new_y, os.path.join(paths, 'correct_impression_word_list.pickle'))