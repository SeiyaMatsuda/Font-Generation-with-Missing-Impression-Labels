import torch
from .mylib import *
from dataset import *
import numpy as np
import torch.nn as nn
import word2vec
SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device ='cuda'
class Font_Generator:
    def __init__(self, G_model, ID, device, data_pararell=True, imp2font=False):
        super().__init__()
        self.ID = ID
        self.G_model = G_model.to(device)
        if data_pararell:
            self.G_model = nn.DataParallel(self.G_model)
        self.device = device
        self.word2vec = word2vec.word2vec()
        self.alpha2num = lambda c: ord(c) -ord('A')
        self.imp2font = imp2font
        if self.imp2font:
            self.z_img = torch.normal(mean=0.5, std=0.2, size=(1000, 300))
        else:
            self.z_img = torch.randn(1000, 256 * 4 * 4)
        self.z_cond = torch.randn(1000, 100)
    def generate_from_impression(self, generate_num, impression_word, alphabet="ABCHERONS", shuffle=False):
        alphabet = list(alphabet)
        alpha_num = list(map(self.alpha2num, alphabet))
        char_num = len(alpha_num)
        char_class = torch.eye(26)[torch.tensor(alpha_num)].repeat(generate_num, 1)
        if shuffle:
            idx = torch.randperm(len(self.z_img))[:generate_num]
        else:
            idx = torch.tensor(list(range(len(self.z_img))))[:generate_num]
        try:
            label = [[self.ID[token] for token in impression_word]]
            label = Multilabel_OneHot(label, len(self.ID), normalize=True)
            emb=True
        except:
            label = [torch.tensor(self.word2vec[ii]) for ii in impression_word]
            label = torch.stack(label)
            label = label.sum(0)
            label = label/(torch.sqrt((label ** 2).sum(0))+ 1e-7)
            label = label.unsqueeze(0)
            emb = False
        label = torch.tensor(label).repeat(char_num * generate_num, 1).to(self.device)
        print(label.shape)
        if self.imp2font:
            noise = self.z_img[idx]
            noise = tile(noise, 0, char_num).to(self.device)
        else:
            z_img = self.z_img[idx]
            z_cond = self.z_cond[idx]
            z_img = tile(z_img, 0, char_num).to(self.device)
            z_cond = tile(z_cond, 0, char_num).to(self.device)
            noise = (z_img, z_cond)
        with torch.no_grad():
            samples = self.G_model(noise, char_class, label, 5, emb=emb)[0]
            samples = samples.data.cpu()
            samples = samples.reshape(-1, char_num, samples.size(2), samples.size(3))
        return samples
    def interpolation_noise(self, word, c=5, alphabet="ABCHERONS"):
        alphabet = list(alphabet)
        alpha_num = list(map(self.alpha2num, alphabet))
        char_num = len(alpha_num)
        char_class = torch.eye(26)[torch.tensor(alpha_num)].repeat(c, 1)
        z_img1 = torch.randn(1, 256 * 4 * 4)
        z_img2= torch.randn(1, 256 * 4 * 4)
        alpha_list = np.linspace(0, 1, c)
        z_img = torch.cat([(1 - alpha) * z_img1 + alpha * z_img2 for alpha in alpha_list])
        z_cond1 = torch.randn(1, 100)
        z_cond2 = torch.randn(1, 100)
        z_cond = torch.cat([(1 - alpha) * z_cond1 + alpha * z_cond2 for alpha in alpha_list])
        z_img = tile(z_img, 0, char_num).to(self.device)
        z_cond = tile(z_cond, 0, char_num).to(self.device)
        label = [[self.ID[token] for token in word]]
        label = Multilabel_OneHot(label, len(self.ID), normalize=True)
        label = torch.tensor(label)
        condition = tile(label, 0, char_num * c).to(self.device)
        noise = (z_img, z_cond)
        print(z_img.shape, z_cond.shape)
        with torch.no_grad():
            samples = self.G_model(noise, char_class, condition, 5)[0]
            samples = samples.data.cpu()
            samples = samples.reshape(-1, char_num, samples.size(2), samples.size(3))
        return samples

    def interpolation_impression(self, word1, word2, c=6, alphabet="ABCHERONS"):
        alphabet = list(alphabet)
        alpha_num = list(map(self.alpha2num, alphabet))
        char_num = len(alpha_num)
        char_class = torch.eye(26)[torch.tensor(alpha_num)].repeat(c, 1)
        z_img = torch.randn(1, 256 * 4 * 4)
        z_cond = torch.randn(1, 100)
        z_img = tile(z_img, 0, char_num * c).to(self.device)
        z_cond = tile(z_cond, 0, char_num * c).to(self.device)
        label1 = [[self.ID[token] for token in word1]]
        label1 = Multilabel_OneHot(label1, len(self.ID), normalize=True)
        label1 = torch.tensor(label1)
        label2 = [[self.ID[token] for token in word2]]
        label2 = Multilabel_OneHot(label2, len(self.ID), normalize=True)
        label2 = torch.tensor(label2)
        alpha_list = np.linspace(0, 1, c)
        condition = torch.cat([(1 - alpha) * label1 + alpha * label2 for alpha in alpha_list])
        condition = tile(condition, 0, char_num).to(self.device)
        noise = (z_img, z_cond)
        with torch.no_grad():
            samples = self.G_model(noise, char_class, condition, 5)[0]
            samples = samples.data.cpu()
            samples = samples.reshape(-1, char_num, samples.size(2), samples.size(3))
        return samples
    
    def generate_random(self, c=100, alphabet="A"):
        alphabet = list(alphabet)
        alpha_num = list(map(self.alpha2num, alphabet))
        char_num = len(alpha_num)
        char_class = torch.eye(26)[torch.tensor(alpha_num)].repeat(c, 1)
        z_img = self.z_img[:c]
        z_cond = self.z_cond[:c]
        label = torch.zeros(c, 1430)
        idx = torch.randint(0, 1430, (c,))
        for i1, i2 in zip(range(len(z_img)), idx):
            label[i1, i2] = 1
        print(label)
        label = label.to(self.device)
        char_class = char_class.to(self.device)
        z_img = z_img.to(self.device)
        z_cond = z_cond.to(self.device)
        noise = (z_img, z_cond)
        print(z_img.shape, z_cond.shape, char_class.shape, label.shape)
        with torch.no_grad():
            samples = self.G_model(noise, char_class, label, 5)[0]
            samples = samples.data.cpu()
            samples = samples.reshape(-1, char_num, samples.size(2), samples.size(3))
        return samples

    def generate_from_changed_ratio(self, label, c=100, alphabet="A", ratio=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], shuffle=False, co_matrix=None):
        alphabet = list(alphabet)
        alpha_num = list(map(self.alpha2num, alphabet))
        char_num = len(alpha_num)
        char_class = torch.eye(26)[torch.tensor(alpha_num)].repeat(c, 1)
        char_class = char_class.to(self.device)
        if shuffle:
            idx = torch.randperm(len(self.z_img))[:c]
        else:
            idx = torch.tensor(range(len(self.z_img)))[:c]
        if self.imp2font:
            noise = self.z_img[idx]
            noise = tile(noise, 0, char_num).to(self.device)
        else:
            z_img = self.z_img[idx]
            z_cond = self.z_cond[idx]
            z_img = tile(z_img, 0, char_num).to(self.device)
            z_cond = tile(z_cond, 0, char_num).to(self.device)
            noise = (z_img, z_cond)
        embed_label = torch.tensor([self.ID[token] for token in label])-1
        if co_matrix!=None:
            imp_num = co_matrix[embed_label, embed_label]
            max_idx = np.argsort(-imp_num)[0]
            co_occurence_score = co_matrix[embed_label[max_idx], embed_label]
            idx = np.argsort(co_occurence_score)
        else:
            idx = torch.randperm(len(embed_label))
        samples_img = []
        samples_label = []
        for r in ratio:
            length = int(len(label)*r)
            label_ = [label[i] for i in idx[:length]]
            samples_label.append(label_)
            label_ = [[self.ID[token] for token in label_]]
            label_ = Multilabel_OneHot(label_, len(self.ID), normalize=True)
            label_ = torch.tensor(label_).repeat(char_num * c, 1).to(self.device)
            label_ = label_.to(self.device)
            with torch.no_grad():
                samples = self.G_model(noise, char_class, label_, 5)[0]
                samples = samples.data.cpu()
                samples = samples.reshape(-1, char_num, samples.size(2), samples.size(3))
            samples_img.append(samples)

        return samples_img, samples_label

