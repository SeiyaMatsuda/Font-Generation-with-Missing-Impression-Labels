from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import gc
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
from torch.nn.functional import adaptive_avg_pool2d
from utils.mylib import Multilabel_OneHot
import numpy as np
import os
from sklearn.model_selection import KFold

import time
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
# print(os.listdir("../input"))
class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

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
        self.X = x.float()# 入力
        self.Y = y # 出力
        self.transform = transform
    def __len__(self):
        return len(self.X) # データ数(10)を返す

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        X = self.X[index]
        if self.transform:
            X = self.transform(X)
        Y = self.Y[index]
        return X, Y

class GAN_train_test(nn.Module):
    def __init__(self, num_class, ID, train_or_test="train", device="cuda"):
        super(GAN_train_test, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize(224)])
        self.num_class = num_class
        self.train_or_test = train_or_test
        self.ID = ID
        self.device = device
        self.save_dir = f"./models/{self.train_or_test}"
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def caliculate_pos_weight(self, label):
        label = self.multi_hot_encoding(label)
        pos_weight = (len(label) - label.sum(0)) / label.sum(0)
        return pos_weight

    def multi_hot_encoding(self, label):
        label = [list(map(lambda x:self.ID[x], l)) for l in label]
        label = Multilabel_OneHot(label, len(self.ID), normalize=False)
        return label

    def build_dataset(self, train_x, train_y, test_x, test_y):
        train_y = self.multi_hot_encoding(train_y)
        test_y = self.multi_hot_encoding(test_y)
        train_dataset = DataSet(train_x, train_y, self.transform)
        test_dataset = DataSet(test_x, test_y, self.transform)
        return train_dataset, test_dataset

    def build_model(self, num_class):
        model = models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(26, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Sequential(
            nn.Linear(2048, num_class)
        )
        return model

    def train(self, inputs, target):
        inputs, target = inputs.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        prediction = self.model(inputs)
        train_loss = self.criterion(prediction.to(self.device), target)
        train_loss.backward()  # 誤差逆伝播
        self.optimizer.step()
        return torch.sigmoid(prediction), train_loss.item()

    def eval(self, inputs, target):
        inputs, target = inputs.to(self.device), target.to(self.device)
        prediction = self.model(inputs)
        eval_loss = self.criterion(prediction.to(self.device), target)
        return torch.sigmoid(prediction), eval_loss.item()

    def run(self, real_img, real_label, fake_img, fake_label, pos_weight, epochs, batch_size=64, shuffle=True, data_pararell=True):
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight).to(self.device)
        if self.train_or_test=="test":
            train_dataset, test_dataset = self.build_dataset(real_img, real_label, fake_img, fake_label)
        elif self.train_or_test=="train":
            train_dataset, test_dataset = self.build_dataset(fake_img, fake_label, real_img, real_label)
        kf = KFold(n_splits=5)
        CV_score = []
        for i, (train_indices, val_indices) in enumerate(kf.split(list(range(len(train_dataset))))):
            self.model = self.build_model(self.num_class).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            earlystopping = EarlyStopping(patience=5, verbose=False, path=os.path.join(self.save_dir, f"'fold{i}.pth'"))
            if data_pararell:
                self.model=nn.DataParallel(self.model)
            tr_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
            train_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=shuffle)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            for epoch in range(epochs):
                self.model.train()
                train_loss = []
                eval_loss = []
                out_ = []
                yy_ = []
                for x_train, y_train in tqdm(train_loader, total=len(train_loader)):
                    prediction, loss = self.train(x_train, y_train)
                    train_loss.append(loss)
                    out_.append(prediction.data.cpu())
                    yy_.append(y_train.data.cpu())
                train_map = mean_average_precision(torch.cat(out_), torch.cat(yy_))
                out_ = []
                yy_ = []
                gc.collect()
                with torch.no_grad():
                    self.model.eval()
                    for x_val, y_val in tqdm(val_loader, total=len(val_loader)):
                        prediction, loss = self.eval(x_val, y_val)
                        eval_loss.append(loss)
                        out_.append(prediction.data.cpu())
                        yy_.append(y_val.data.cpu())
                eval_map = mean_average_precision(torch.cat(out_), torch.cat(yy_))
                total_train_loss = np.asarray(train_loss).mean()
                total_train_map = np.asarray(train_map).mean()
                total_val_loss = np.asarray(eval_loss).mean()
                total_val_map = np.asarray(eval_map).mean()
                print(f"Epoch: {epoch}, train_loss: {total_train_loss}.")
                print(f"Epoch: {epoch}, train_accuracy: {total_train_map}.")
                print(f"Epoch: {epoch}, val_loss: {total_val_loss}.")
                print(f"Epoch: {epoch}, val_map: {total_val_map}.")
                earlystopping(total_val_loss, self.model)  # callメソッド呼び出し
                if earlystopping.early_stop:  # ストップフラグがTrueの場合、breakでforループを抜ける
                    print("Early Stopping!")
                    break
            out_ = []
            yy_ = []
            with torch.no_grad():
                self.model.eval()
                for x_test, y_test in test_loader:
                    pred_test, _ = self.eval(x_test, y_test)
                    out_.append(pred_test)
                    yy_.append(y_test)
            test_map = mean_average_precision(torch.cat(out_), torch.cat(yy_))
            score = np.asarray(test_map).mean()
            print((f"fold{i}, test_map: {score}."))
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

def ranking_acc(pred, true):
    pred_rank = np.argsort(-pred)
    result = []
    for i in tqdm.tqdm(range(1574), total = 1574):
        ranking = pred_rank[:,:i+1]
        target = np.tile(true, (i+1,1)).T
        result.append(((ranking == target).sum(1)>0).sum()/ranking.shape[0])
    return result

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
        average_precisions.append(precision[mask].mean().item())
    AP = [x for x in average_precisions if np.isnan(x) == False]
    return sum(AP)/len(y_true), average_precisions
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
    real_img = torch.zeros(size=(len(label), 26, 64, 64)).float()
    fake_img = torch.zeros(size=(len(label), 26, 64, 64)).float()
    a = GAN_train_test(len(ID), ID, "train", 'cuda')
    pos_weight = a.caliculate_pos_weight(label)
    a.run(fake_img.reshape(-1, 26, fake_img.size(2), fake_img.size(3)), label, real_img.reshape(-1, 26, real_img.size(2), real_img.size(3)), label, pos_weight, 64, batch_size=64,
          shuffle=True)