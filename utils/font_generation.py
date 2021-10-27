from .mylib import *
from dataset import *
import numpy as np
import torch.nn as nn
SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device ='cuda'
class Font_Generator:
    def __init__(self, G_model, path, ID, device, data_pararell=True):
        super().__init__()
        self.ID = ID
        self.G_model = G_model.to(device)
        if data_pararell:
            self.G_model = nn.DataParallel(self.G_model)
        self.weight = torch.load(path)["G_net"]
        self.G_model.load_state_dict(torch.load(path)["G_net"], strict=False)
        self.device = device
        self.alpha2num = lambda c: ord(c) -ord('A')
    def generate_from_impression(self, generate_num, impression_word, alphabet="ABCHERONS"):
        alphabet = list(alphabet)
        alpha_num = list(map(self.alpha2num, alphabet))
        char_num = len(alpha_num)
        char_class = torch.eye(26)[torch.tensor(alpha_num)].repeat(generate_num, 1)
        z_img = torch.randn(generate_num, 256 * 4 * 4)
        z_cond = torch.randn(generate_num, 100)
        z_img = tile(z_img, 0, char_num).to(device)
        z_cond = tile(z_cond, 0, char_num).to(device)
        label = [[self.ID[token] for token in impression_word]]
        label = Multilabel_OneHot(label, len(self.ID), normalize=True)
        label = torch.tensor(label).repeat(char_num * generate_num, 1).to(self.device)
        noise = (z_img, z_cond)
        with torch.no_grad():
            samples = self.G_model(noise, char_class, label, 5)[0]
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
        z_cond = torch.randn(1, 100)
        z_img = tile(z_img, 0, char_num).to(device)
        z_cond = tile(z_cond, 0, char_num * c).to(device)
        label = [[self.ID[token] for token in word]]
        label = Multilabel_OneHot(label, len(self.ID), normalize=True)
        label = torch.tensor(label)
        condition = tile(label, 0, char_num * c).to(device)
        noise = (z_img, z_cond)
        with torch.no_grad():
            samples = self.G_model(noise, char_class, condition, 5)[0]
            samples = samples.data.cpu()
            samples = samples.reshape(-1, char_num, samples.size(2), samples.size(3))
        return samples

    def interpolation_impression(self, word1, word2, c=5, alphabet="ABCHERONS"):
        alphabet = list(alphabet)
        alpha_num = list(map(self.alpha2num, alphabet))
        char_num = len(alpha_num)
        char_class = torch.eye(26)[torch.tensor(alpha_num)].repeat(c, 1)
        z_img = torch.randn(1, 256 * 4 * 4)
        z_cond = torch.randn(1, 100)
        z_img = tile(z_img, 0, char_num * c).to(device)
        z_cond = tile(z_cond, 0, char_num * c).to(device)
        label1 = [[self.ID[token] for token in word1]]
        label1 = Multilabel_OneHot(label1, len(self.ID), normalize=True)
        label1 = torch.tensor(label1)
        label2 = [[self.ID[token] for token in word2]]
        label2 = Multilabel_OneHot(label2, len(self.ID), normalize=True)
        label2 = torch.tensor(label2)
        alpha_list = np.linspace(0, 1, c)
        condition = torch.cat([(1 - alpha) * label1 + alpha * label2 for alpha in alpha_list])
        condition = tile(condition, 0, char_num).to(device)
        noise = (z_img, z_cond)
        with torch.no_grad():
            samples = self.G_model(noise, char_class, condition, 5)[0]
            samples = samples.data.cpu()
            samples = samples.reshape(-1, char_num, samples.size(2), samples.size(3))
        return samples