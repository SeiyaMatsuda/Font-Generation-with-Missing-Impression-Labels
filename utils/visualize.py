import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from .mylib import tile
from sklearn.decomposition import PCA
def visualizer(path, G_model, z, char_num, label, res, device, nibuchan):
    G_model.eval()
    z_img = z[0]
    z_cond = z[1]
    z_shape = z_img.shape
    label_shape = label.shape
    char_cl = torch.tensor([0, 1, 2, 7, 4, 17, 14, 13, 18])
    if nibuchan:
        char_cl = torch.tensor([13, 8, 1, 20, 0, 10, 0, 17, 8])
    char = torch.eye(char_num)[char_cl].repeat(z_shape[0] * label_shape[0], 1).to(device)
    char_num = len(char_cl)
    z_img = tile(z_img, 0, char_num).repeat(label_shape[0], 1).to(device)
    z_cond = tile(z_cond, 0, char_num).repeat(label_shape[0], 1).to(device)
    label = tile(label, 0, char_num * z_shape[0]).to(device)
    z = (z_img, z_cond)
    with torch.no_grad():
        samples = G_model(z, char, label, res)[0].data.cpu()
        samples = F.interpolate(samples, (128, 128), mode='nearest')
        samples = samples/2 + 0.5
        save_image(samples, path, nrow=char_num)

def learning_curve(dict, path, title ='learning_curve', x_label = 'epoch', y_label = 'loss'):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Traing score と Test score をプロット
    for key, value in dict.items():
        plt.plot(range(len(value)), np.array(value),  label=str(key))
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()

def imscatter(x, y, data, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    artists = []
    for x0, y0, d in zip(x, y, data):
        im = OffsetImage(d, cmap = plt.cm.gray_r, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    return artists
class visualize_semantic_condition:
    def __init__(self, weight):
        self.pca = PCA(n_components=2)
        weight = torch.tensor(weight)
        self.weight = weight/(torch.linalg.norm(weight, dim=1).unsqueeze(1) + 1e-7)
    def visualize_sc_w2v(self, image, attr):
        x_, y_ = self.pca.fit_transform(self.weight).T
        x, y = self.pca.transform(attr.data.cpu()).T
        fig, ax = plt.subplots(figsize=(20.0, 20.0))
        imscatter(x, y, 255 - image[:, 0, :, :].to('cpu').detach().numpy().copy(), ax=ax,  zoom=.25)
        ax.plot(x, y, 'o', alpha=0)
        ax.autoscale()
        ax.plot(x_, y_, 'o', color='green', markersize=10,  alpha=0.4)
        return fig
    def visualize_sc(self, attr1, attr2):
        self.pca.fit(self.weight)
        x1, y1 = self.pca.transform(attr1.data.cpu()).T
        x2, y2 = self.pca.transform(attr2.data.cpu()).T
        fig, ax = plt.subplots(figsize=(20.0, 20.0))
        ax.plot(x1, y1, 'o', color='red', markersize=10,)
        ax.plot(x2, y2, 'o', color='green', markersize=10,)
        ax.autoscale()
        return fig
