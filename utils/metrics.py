from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import matplotlib.animation as animation
from IPython.display import HTML
import cv2
from scipy import linalg
import tqdm
from torch.nn.functional import adaptive_avg_pool2d
from utils.mylib import Multilabel_OneHot
import numpy as np
import os
from sklearn.model_selection import KFold

import time
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

class FID(nn.Module):
    def __init__(self):
        super(FID, self).__init__()
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx])
        self.model = self.model.cuda()

    def calculate_activation_statistics(self, images, model, batch_size=128, dims=2048, verbose = True,
                                        cuda=False):
        model.eval()

        if len(images) % batch_size != 0:
            print(('Warning: number of images is not a multiple of the '
                   'batch size. Some samples are going to be ignored.'))
        if batch_size > len(images):
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = len(images)
        # print(batch_size)
        n_batches = len(images) // batch_size
        n_used_imgs = n_batches * batch_size

        act = np.empty((n_used_imgs, dims))

        for i in tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                      end='', flush=True)
            start = i * batch_size
            end = start + batch_size

            if cuda:
                batch = images[start:end].cuda()
            else:
                batch = images[start:end]
            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose FID dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

        if verbose:
            print(' done')

        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


    def calculate_fretchet(self, images_real, images_fake, cuda=True, verbose=False, batch_size=128):
        mu_1, std_1 = self.calculate_activation_statistics(images_real, self.model, cuda=cuda, verbose=verbose, batch_size=batch_size)
        mu_2, std_2 = self.calculate_activation_statistics(images_fake, self.model, cuda=cuda, verbose=verbose, batch_size=batch_size)

        """get fretched distance"""
        fid_value = self.calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
        return fid_value
class DataSet(torch.utils.data.Dataset):
    def __init__(self, x, y, transform = None):
        self.X = x * 255# 入力
        self.Y = y # 出力
        self.transform = transform
    def __len__(self):
        return len(self.X) # データ数(10)を返す

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        X = self.X[index]
        if self.transform:
            X = self.transform(X)/255
        Y = self.Y[index]
        return X, Y

class GAN_train_test(nn.Module):
    def __init__(self, num_class, ID, device="cuda"):
        super(GAN_train_test, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor()])
        self.model = self.build_model(num_class).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.num_class = num_class
        self.ID = ID
        self.device = device

    def caliculate_pos_weight(self, label):
        label = self.multi_hot_encoding(label)
        pos_weight = (len(label) - label.sum(0)) / label.sum(0)
        return pos_weight

    def multi_hot_encoding(self, label):
        label = [list(map(lambda x:self.ID[x], l)) for l in label]
        label = Multilabel_OneHot(label, len(self.ID), normalize=False)
        return label

    def build_dataset(self, train_x, train_y, test_x, test_y, num_class):
        train_y = self.multi_hot_encoding(train_y)
        test_y = self.multi_hot_encoding(test_y)
        train_dataset = DataSet(train_x, train_y, self.transform)
        test_dataset = DataSet(test_x, test_y, self.transform)
        return train_dataset, test_dataset

    def build_model(self, num_class):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(512, num_class)
        )
        return model

    def train(self, inputs, target):
        inputs, target = inputs.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        prediction = self.model(inputs)
        train_map = mean_average_precision(prediction.data.cpu(), target.data.cpu())
        train_loss = self.criterion(prediction.to(self.device), target)
        train_loss.backward()  # 誤差逆伝播
        self.optimizer.step()
        return train_map, train_loss.item()

    def eval(self, inputs, target):
        prediction = self.model(inputs)
        eval_map = mean_average_precision(prediction.data.cpu(), target.data.cpu())
        eval_loss = self.criterion(prediction.data.cpu(), target.data.cpu()).item()
        return eval_map, eval_loss

    def run(self, train_X, train_y, test_X, test_y, pos_weight, epochs, batch_size=64, shuffle=True):
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight).to(self.device)
        train_dataset, test_dataset = self.build_dataset(train_X, train_y, test_X, test_y, num_class=1430)
        kf = KFold(n_splits=5)
        CV_score = []
        for train_indices, val_indices in kf.split(list(range(len(train_dataset)))):
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
            for epoch in range(epochs):
                self.model.train()
                train_loss = []
                train_map = []
                eval_loss = []
                eval_map = []
                for x_train, y_train in tqdm.tqdm(train_loader, total=len(train_loader)):
                    map, loss = self.train(x_train, y_train)
                    train_loss.append(loss)
                    train_map.append(map)
                self.model.eval()
                for x_val, y_val in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
                    map, loss = self.eval(x_val, y_val)
                    eval_loss.append(loss)
                    eval_map.append(map)
                total_train_loss = np.asarray(train_loss).mean()
                total_train_map = np.asarray(train_map).mean()
                total_val_loss = np.asarray(eval_loss).mean()
                total_val_map = np.asarray(eval_map).mean()
                print(f"Epoch: {epoch}, loss: {total_train_loss}.")
                print(f"Epoch: {epoch}, accuracy: {total_train_map}.")
                print(f"Epoch: {epoch}, accuracy: {total_val_loss}.")
                print(f"Epoch: {epoch}, accuracy: {total_val_map}.")
            test_map = []
            for x_test, y_test in enumerate(test_loader, total=len(test_loader)):
                test_map, _ = self.eval(x_test, y_test)
                test_map.append(test_map)
            score = np.asarray(test_map).mean()
            CV_score.append(score)
        return CV_score

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

if __name__ == '__main__':
    import sys
    from options import get_parser
    sys.path.append(os.path.join(os.path.dirname("__file__"), "../../"))
    print(sys.path)
    parser = get_parser()
    opts = parser.parse_args(args=[])
    data = torch.from_numpy(np.array([np.load(d) for d in opts.data]))[:, :26, :, :].float() / 127.5 - 1
    label = opts.impression_word_list
    ID = {key: idx + 1 for idx, key in enumerate(opts.w2v_vocab)}
    real_img = torch.zeros(size=(len(label), 1, 64, 64))
    fake_img = torch.zeros(size=(len(label), 1, 64, 64))
    a = GAN_train_test(len(ID), ID, 'cuda')
    pos_weight = a.caliculate_pos_weight(label)
    a.run(fake_img.expand(-1, 3, -1, -1), label, real_img.expand(-1, 3, -1, -1), label, pos_weight, 64, batch_size=64,
          shuffle=True)