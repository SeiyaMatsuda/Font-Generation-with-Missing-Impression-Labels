import pickle
import random
import tqdm
import torch
import operator
import cv2
import glob
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import bernoulli
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch.autograd as autograd
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
def pickle_dump(obj,path):
    with open(path,mode="wb") as f:
        pickle.dump(obj, f)
def pickle_load(path):
    with open(path,mode="rb") as f:
        data=pickle.load(f)
        return data

def return_index(label,weight):
    dice = list(range(len(label)))
    # 6の目が出やすいように重みを設定する
    if len(weight) == 0:
        samples = random.choices(dice)
    else:
        samples = random.choices(dice, k=1, weights=[1 /w **2  for w in weight])
    return samples
def tile_like(x, img):
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = x.repeat(1, 1, img.size(2), img.size(3))
    return x
def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm.tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def split_list(l, n):
    for idx in range(0,len(l),n):
        yield l[idx:idx+n]

def Multilabel_OneHot(labels, n_categories, dtype=torch.float32, normalize=True):
    batch_size = len(labels)
    one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype)
    for i, label in enumerate(labels):
        # Subtract 1 from each LongTensor because your
        # indexing starts at 1 and tensor indexing starts at 0
        label = torch.LongTensor(label)-1
        one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
        
    if normalize:
        return torch.mul(one_hot_labels, 1/one_hot_labels.sum(axis = 1).view(-1, 1))
    else:
        return one_hot_labels
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)
def split_list(l, n):
    """
    リストをサブリストに分割する
    :param l: リスト
    :param n: サブリストの要素数
    :return: 
    """
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]

def missing2clean(prob):
    sampler = torch.distributions.categorical.Categorical(prob)
    clean_label = sampler.sample()
    clean_hot = torch.eye(prob.size(1))[clean_label]
    return clean_hot

# 共起行列作成関数の実装
def create_co_matrix(label, ID):
    # 総単語数を取得
    corpus_size = len(ID)

    # 共起行列を初期化
    co_matrix = torch.zeros((corpus_size, corpus_size))

    # 1語ずつ処理
    for ll in label:
        ll = list(map(lambda x: ID[x] - 1, ll))
        one_hot_labels = torch.zeros(size=(corpus_size,))
        one_hot_labels.scatter_(dim=0, index=torch.LongTensor(ll), value=1.)
        for idx in ll:
            co_matrix[idx] += one_hot_labels
    return co_matrix

def missing2prob(input, co_matrix):
    co_matrix = co_matrix/(co_matrix.sum(0)+1e-7)
    output = torch.mm(input, co_matrix.T)
    return output

def missing2prob(input, co_matrix):
    co_matrix_n = co_matrix/(np.diag(co_matrix))
    output = torch.mm(input, co_matrix_n.T)/input.sum(1).unsqueeze(1)
    output[input==1]=1
    return output
