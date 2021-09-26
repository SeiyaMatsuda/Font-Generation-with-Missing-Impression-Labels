import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from .mylib import tile
def visualizer(path, G_model, z, char_num, label, res, device):
    G_model.eval()
    z_img = z[0]
    z_cond = z[1]
    z_shape = z_img.shape
    label_shape = label.shape
    char = torch.eye(char_num).repeat(z_shape[0] * label_shape[0], 1).to(device)
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