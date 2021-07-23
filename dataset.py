import torch
import numpy as np
from mylib import return_index
from options import get_parser
import os
import itertools
import word2vec
import torchvision.transforms as transforms
import glob
from sklearn.preprocessing import LabelEncoder
import random
import tqdm
from collections import Counter
import torchvision
from PIL import Image
def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if elem_type == np.ndarray:
        try:
            return torch.Tensor(np.stack(batch))
        except ValueError:
            return batch
    elif elem_type == torch.Tensor:
        try:
            return torch.stack(batch)
        except RuntimeError:
            return batch
    elif elem_type == list:
        return batch
    elif elem_type == int:
        return torch.LongTensor(batch)
    elif elem_type == str:
        return batch
    elif elem_type == dict:
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
class Transform(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return (sample/127.5)-1

class Myfont_dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, ID,  transform=None, char_num = 52, n_style = 4, img_size = 64):
        self.transform = transform
        self.data = []
        self.char_class = []
        self.dataset = []
        self.char_num = char_num
        self.n_style = n_style
        self.img_size = img_size
        self.ID = ID
        self.char_idx_offset = 0
        self.chars = list(range(self.char_idx_offset, self.char_idx_offset+self.char_num))
        self.data_num = len(label)
        self.count1 = 0
        self.weight = dict(Counter(sum(label, [])))
        self.weight = torch.tensor([self.weight[key] if key in self.weight.keys() else 0 for key in self.ID.keys()]).float()
        for idx, ll in enumerate(label):
            if len(ll)==0:
                continue
            self.multi_label = ll
            self.multi_embed_label = list(map(lambda x: ID[x], self.multi_label))
            self.one_label = random.choices(ll, k = char_num)
            self.one_embed_label = list(map(lambda x:ID[x], self.one_label))
            for j in range(char_num):
                self.data = data[idx][j].astype(np.float32).reshape(-1, self.img_size, self.img_size)
                self.char_class = j
                self.dataset.append([self.data, self.one_label[j], self.multi_label, self.char_class, self.one_embed_label[j], self.multi_embed_label])
        self.data_num = len(self.dataset)
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        img, one_label,  multi_label, \
        charclass, one_embed_label, multi_embed_label \
            = self.dataset[idx]

        return {"img": self.transform(img),
                "one_label": one_label,
                "multi_label": multi_label,
                "charclass": charclass,
                "one_embed_label": one_embed_label,
                'multi_embed_label': multi_embed_label
                }


    def weight(self):
        self.w = torch.Tensor(self.w)
        return self.w

    def inverse_transform(self,a):
        return list(self.le.inverse_transform(a))

class Myfont_dataset2(torch.utils.data.Dataset):
    def __init__(self, data, label, ID,  transform=None, char_num = 52, n_style = 4, img_size = 64, binary = False):
        self.transform = transform
        self.data = []
        self.one_label = []
        self.char_class = []
        self.target_dataset = []
        self.source_dataset = []
        self.char_num = char_num
        self.n_style = n_style
        self.img_size = img_size
        self.ID = ID
        self.char_idx_offset = 0
        self.chars = list(range(self.char_idx_offset, self.char_idx_offset+self.char_num))
        self.prob = {}
        self.data_num = len(label)
        # 特定のラベルが選ばれる期待値を算出
        for i in range(self.data_num):
            for key in label[i]:
                if key not in self.prob.keys():
                    self.prob[key] = (1 / self.data_num) * (1 / len(label[i]))
                else:
                    self.prob[key] = self.prob[key] + (1 / self.data_num) * (1 / len(label[i]))
        for i in tqdm.tqdm(range(len(data)),total = len(data)):
            if len(label[i]) != 0:
                tag_prob = []
                for key in label[i]:
                    tag_prob.append(self.prob[key])
                r = return_index(label[i], tag_prob)
                self.one_label = label[i][r[0]]
                self.multi_label = label[i]
                self.one_embed_label = self.ID[self.one_label]
                self.multi_emebed_label = [self.ID[key] for key in self.multi_label]
                for j in range(self.char_num):
                    self.data = data[i][j].astype(np.float32).reshape(-1, self.img_size, self.img_size)
                    self.char_class = j
                    self.target_dataset.append([self.data, self.one_label, self.multi_label, self.char_class, self.one_embed_label, self.multi_emebed_label])
                    if 'plain' in label[i]:
                        self.source_dataset.append([self.data, self.char_class])
            else:
                continue
        self.weight = dict(Counter(sum(label, [])))
        self.weight = torch.tensor([self.weight[key] if key in self.weight.keys() else 0 for key in self.ID.keys()]).float()
        self.data_num = len(self.target_dataset)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        img_target, one_label_target, multi_label_target, \
        charclass_target, one_embed_label_target, multi_embed_label_target\
            = self.target_dataset[idx]
        idx_source = idx % self.char_num + self.char_num * random.randint(0, int((len(self.source_dataset)/self.char_num) - 1))
        img_source, charclass_source = self.source_dataset[idx_source]
        # Get style samples
        random.shuffle(self.chars)
        style_chars = self.chars[:self.n_style]
        style_imgs_target = []
        styles_index = list(map(lambda x: x+idx-(idx % self.char_num),style_chars))
        for char in styles_index:
            style_imgs_target.append(self.transform(self.target_dataset[char][0]))

        random.shuffle(self.chars)
        style_chars = self.chars[:self.n_style]
        style_imgs_source = []
        styles_index = map(lambda x: x+idx_source-(idx_source % self.char_num),style_chars)
        for char in styles_index:
            style_imgs_source.append(self.transform(self.source_dataset[char][0]))

        style_imgs_target = np.concatenate(style_imgs_target)
        style_imgs_source = np.concatenate(style_imgs_source)

        return {"img_target": self.transform(img_target),
                "one_label_target": one_label_target, "multi_label_target": multi_label_target,
                "charclass_target": charclass_target,
                "one_embed_label_target": one_embed_label_target, "multi_embed_label_target": multi_embed_label_target,
                "styles_target": style_imgs_target,
                "img_source": self.transform(img_source), "charclass_source": charclass_source,
                "styles_source": style_imgs_source
                }


    def weight(self):
        self.w = torch.Tensor(self.w)
        return self.w

    def inverse_transform(self,a):
        return list(self.le.inverse_transform(a))


class Myfont_dataset3(torch.utils.data.Dataset):
    def __init__(self, data, label, ID,  transform=None, char_num = 52, n_style = 4, img_size = 64):
        self.transform = transform
        self.data = []
        self.char_class = []
        self.dataset = []
        self.char_num = char_num
        self.n_style = n_style
        self.img_size = img_size
        self.ID = ID
        self.char_idx_offset = 0
        self.chars = list(range(self.char_idx_offset, self.char_idx_offset+self.char_num))
        self.data_num = len(label)
        self.count1 = 0
        self.weight = dict(Counter(sum(label, [])))
        self.weight = torch.tensor([self.weight[key] if key in self.weight.keys() else 0 for key in self.ID.keys()]).float()
        for key, value in tqdm.tqdm(ID.items(), total=len(ID)):
            self.label = key
            self.embed_label = value
            idx = [idx for idx, ll in enumerate(label) if key in ll]
            idx = random.choices(idx, k=10)
            for i in idx:
                for j in range(char_num):
                    self.data = data[i][j].astype(np.float32).reshape(-1, self.img_size, self.img_size)
                    self.char_class = j
                    self.dataset.append([self.data, self.label, self.char_class, self.embed_label])
        self.data_num = len(self.dataset)
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        img, label, charclass, embed_label \
            = self.dataset[idx]

        return {"img": self.transform(img),
                "label": label,
                "charclass": charclass,
                "embed_label": embed_label
                }


    def weight(self):
        self.w = torch.Tensor(self.w)
        return self.w

    def inverse_transform(self,a):
        return list(self.le.inverse_transform(a))
if __name__=="__main__":
    parser = get_parser()
    opts = parser.parse_args()
    data = np.array([np.load(d) for d in opts.data])
    transform = transforms.Compose(
        [Transform()])
    ID = {}
    ID = {key: idx + 1 for idx, key in enumerate(opts.w2v_vocab)}
    data = Myfont_dataset3(data, opts.correct_impression_word_list, ID, char_num = 26, transform=transform)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True, collate_fn = collate_fn)
    for t in tqdm.tqdm(dataloader,total= len(dataloader)):
        exit()
