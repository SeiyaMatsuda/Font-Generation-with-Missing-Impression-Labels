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
def label_preprocess(text):
    global tag_vectors
    text=text.replace("-", " ")
    tokens=text.split()
    return tokens
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

def compute_gradient_penalty(D, real_samples, fake_samples, Tensor, label = None, char_class = None):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[0].view(-1, 1)
    fake = Tensor(d_interpolates.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    del gradients, d_interpolates
    return gradient_penalty
def Generate_img(epoch, G_model, ID, device, source_img, noise, label, log_dir):
    label = Multilabel_OneHot(label, len(ID))
    G_model.eval()
    with torch.no_grad():
        source_img = source_img
        # source_img = torch.cat([tensor.expand(len(label), z_dim) for tensor in torch.chunk(source_img, 5, dim=0)], axis=0)
        #Generatorでサンプル生成
        noise = noise.expand(len(source_img), -1, -1, -1)
        label = label.to(device)
        samples = G_model(noise, source_img, label)[0].data.cpu()
        samples = torch.cat([source_img[0:26].data.cpu(), (samples/2)+0.5], dim = 0)
        save_image(samples,os.path.join(log_dir, 'epoch_%05d.png' % (epoch)), nrow = 26)
def loss_plot(log_dir,D_loss_list,G_loss_list):
    fig = plt.figure(figsize=(10,7))
    loss = fig.add_subplot(1,1,1)
    loss.plot(range(len(D_loss_list)),D_loss_list,label='Discriminator_loss')
    loss.plot(range(len(G_loss_list)),G_loss_list,label='Generator_loss')
    loss.set_xlabel('epoch')
    loss.set_ylabel('loss')
    loss.legend()
    loss.grid()
    fig.show()
    plt.show()
    fig.savefig(os.path.join(log_dir,"train_loss.png"))
def make_randomwalk(log_dir):
    files = sorted(glob.glob(os.path.join(log_dir,'logs_cWGAN/epoch_*.png')))
    images = list(map(lambda file : Image.open(file) , files))
    images[0].save(os.path.join(log_dir,'randomwalk.gif') , save_all = True , append_images = images , duration = 100 , loop = 0)
def split_list(l, n):
    for idx in range(0,len(l),n):
        yield l[idx:idx+n]
def pseudo_hamming(v1, v2):
    start_time=time.time()
    v1 = v1.reshape(-1,64,64)
    v2 = v2.reshape(-1,64,64)
    bin_img1 = np.where(v1 > 127, 255, 0).astype(np.uint8)
    bin_img2 = np.where(v2 > 127, 255, 0).astype(np.uint8)
    mask_img1 = np.where(bin_img1 > 127, 0, 1).astype(np.uint8)
    mask_img2 = np.where(bin_img2 > 127, 0, 1).astype(np.uint8)
    dist_img1 = np.array([cv2.distanceTransform(b1, cv2.DIST_L2, 3) for b1 in bin_img1])
    dist_img2 = np.array([cv2.distanceTransform(b2, cv2.DIST_L2, 3) for b2 in bin_img2])
    masked_dist_img1 = np.multiply(dist_img1, mask_img2)
    masked_dist_img2 = np.multiply(dist_img2, mask_img1)
    merged_masked_dist_img = masked_dist_img1 + masked_dist_img2
    total = np.sum(merged_masked_dist_img)
    end_time =time.time()
    return total
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
def mean_average_precision(y_pred, y_true):
    average_precisions = []
    # クラス単位でAPを計算
    y_true = y_true.T
    y_pred = y_pred.T
    for i in range(len(y_true)):
        sort_idx = torch.argsort(y_pred[i], descending=True)
        y_true_sorted = y_true[i][sort_idx]
        cumsum = torch.cumsum(y_true_sorted, dim = 0)
        precision = cumsum / torch.arange(1, 1 + y_true[i].shape[0])
        # 代表点
        mask = (y_true_sorted==1)
        average_precisions.append(precision[mask].mean())
    return sum(average_precisions)/len(y_true)
def kl_divergence(input, target, activation = None):
    entropy = -(target[target != 0] * target[target != 0].log()).sum()
    if activation == 'softmax':
        cross_entropy = -(target * F.log_softmax(input, dim=1)).sum()
    elif activation == 'sigmoid':
        cross_entropy = -(target * F.logsigmoid(input)).sum()
    else:
        cross_entropy = -(target * input).sum()
    return (cross_entropy - entropy) / input.size(0)
def imscatter(x, y, image_list, ax=None, zoom=0.2, color='black'):
    for i in range(len(image_list)):
        if ax is None:
            ax = plt.gca()

        image = image_list[i]  # plt.imread(image_list[i])
        im = OffsetImage(image, zoom=zoom)
        artists = []
        x0 = x[i]
        y0 = y[i]
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False, bboxprops=dict(color=color))
        artists.append(ax.add_artist(ab))
    return artists
def Multilabel_OneHot(labels, n_categories, dtype=torch.float32, normalize = True):
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
def extract(G_model, target, inputs):
    feature = None

    def forward_hook(module, inputs, outputs):
        # 順伝搬の出力を features というグローバル変数に記録する
        global features
        features = outputs.detach()

    # コールバック関数を登録する。
    handle = target.register_forward_hook(forward_hook)

    # 推論する
    G_model.eval()
    G_model(inputs)

    # コールバック関数を解除する。
    handle.remove()

    return features
class CXLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCXHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        # [0] means get the value, torch min will return the index as well
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''

        # print("featureT target size:", featureT.shape)
        # print("featureI inference size:", featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=3)[0].max(dim=2)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX
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

def visualizer(path, G_model, z, char_num, label, res, device):
    G_model.eval()
    z_shape = z.shape
    label_shape = label.shape
    char = torch.eye(char_num).repeat(z_shape[0] * label_shape[0], 1).to(device)
    z = tile(z, 0, char_num).repeat(label_shape[0], 1).to(device)
    label = tile(label, 0, char_num * z_shape[0]).to(device)
    with torch.no_grad():
        samples = G_model(z, char, label, res)[0].data.cpu()
        samples = F.interpolate(samples, (128, 128), mode='nearest')
        # samples = samples/2 + 0.5
        save_image(samples, path, nrow=char_num)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        target = target.float()
        # BCELossWithLogits
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, eps=1e-7):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#
#     def forward(self, input, target):
#         logit = F.softmax(input, dim=1)
#         logit = logit.clamp(self.eps, 1. - self.eps)
#         logit_ls = torch.log(logit)
#         loss = F.nll_loss(logit_ls, target, reduction="none")
#         view = target.size() + (1,)
#         index = target.view(*view)
#         loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss
#
#         return loss.sum()

class KlLoss(nn.Module):
    def __init__(self, activation = None):
        super(KlLoss, self).__init__()
        self.activation = activation
        self.eps = 1 * 1e-7
    def forward(self, input, target):
        entropy = -(((target[target != 0] * target[target != 0]) + self.eps).log()).sum()
        if self.activation == 'softmax':
            cross_entropy = -(target * F.log_softmax(input + self.eps, dim=1)).sum()
        elif self.activation == 'sigmoid':
            cross_entropy = -(target * F.logsigmoid(input + self.eps)).sum()
        else:
            cross_entropy = -(target * input).sum()
        return (cross_entropy - entropy) / (input.size(0))

if __name__ == '__main__':
    loss = FocalLoss()
    input  = torch.tensor([0.2,0.5, 0.7, 0.21, 0.4]).view(1, -1)
    target= torch.tensor([0, 1, 0, 1, 0]).view(1, -1)
    loss(input, target)