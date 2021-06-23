from models.DCmodel import ACGenerator, CGenerator
from options import get_parser
from mylib import *
import word2vec
import torch
import random
from dataset import *
import numpy as np
import numpy as np
import torch
import tensorflow as tf
import gc
from models.DCmodel import ACGenerator, CGenerator
from options import get_parser
import word2vec
from dataset import *
from tqdm import trange
from sklearn.model_selection import train_test_split
import random
import tqdm
import pprint
import torch.optim as optim
import numpy as np
from  mylib import *
from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
from stealthflow.fid import FIDNumpy, FIDTF
device = 'cuda'
import torch.nn as nn
CGAN_path = '/home/matsuda/デスクトップ/Imp2Font/experiment/CGAN/model_100'
CPGAN_path = '/home/matsuda/デスクトップ/Imp2Font/experiment/CPGAN/model_50'
ACGAN_path = '/home/matsuda/デスクトップ/Imp2Font/experiment/ACGAN/model_50'
Imp2Font_path = '/home/matsuda/デスクトップ/Imp2Font/experiment/Imp2Font/model_60'
parser = get_parser()
opts = parser.parse_args()
data = np.array([np.load(d) for d in opts.data])
var_mode = False  # 表示結果を見るときに毎回異なる乱数を使うかどうか
#単語IDの変換
trained_embed_model = word2vec.word2vec()
word_vectors = trained_embed_model
weights = word_vectors.vectors
ID = {}
c = 1
mask = []
for idx, key in enumerate(trained_embed_model.vocab.keys()):
    if key in opts.w2v_vocab.keys():
        ID[key] = c
        c+=1
        mask.append(idx)
    else:
        continue
transform= Transform()
gen = 'multi'
if gen == 'single':
    dataset = Myfont_dataset(data, opts.correct_impression_word_list, ID, char_num = opts.char_num, transform=transform)
elif gen == 'multi':
    dataset = Myfont_dataset2(data, opts.correct_impression_word_list, ID, char_num=opts.char_num, transform=transform)
counter = dataset.weight
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.char_num, shuffle=False, collate_fn = collate_fn)
rare_index = list(np.where(np.array(counter)<=15)[0])
class DataSet(torch.utils.data.Dataset):
    def __init__(self, set, transform=None):
        self.X = (set[0] * 255).astype(np.uint8)  # 入力
        self.Y = set[1]  # 出力
        self.X = self.X.reshape(-1 , self.X.shape[1], self.X.shape[2], opts.char_num)
        self.Y = self.Y.reshape(-1, self.Y.shape[1], opts.char_num)[:, :, 0]
        self.transform = transform

    def __len__(self):
        return len(self.X)  # データ数(10)を返す

    def __getitem__(self, index):
        # index番目の入出力ペアを返す
        X = self.X[index]
        if self.transform:
            X = self.transform(X) / 255
        Y = self.Y[index]
        return X, Y

def generator(model_path , dataloader, mode = opts.mode):
    if gen =='single':
        if mode=='AC' or mode == 'CP':
            G_model = ACGenerator(weights, mask, z_dim=300, char_num=26, mode=mode, emb=opts.emb).to('cuda')
        elif mode=='C':
            G_model = CGenerator(weights, mask, z_dim=300, char_num=26, mode=mode, emb=opts.emb).to('cuda')
        G_model = nn.DataParallel(G_model)
        G_model.load_state_dict(torch.load(model_path)["G_model_state_dict"], strict=False)
        G_model.eval()
        real_imgs = []
        fake_imgs = []
        char_class = []
        labels = []
        for iter, data in tqdm.tqdm(enumerate(dataloader),  total=len(dataloader)):
            batch_size = len(data['img'])
            label, charclass = data['embed_label'], data['charclass']
            noise =  torch.normal(mean = 0.5, std = 0.2, size = (batch_size, 300))
            charclass = torch.eye(26)[charclass]
            condition = torch.eye(len(ID))[label-1]
            with torch.no_grad():
                samples = G_model(noise, condition, charclass)
                samples = samples.data.cpu()
                samples = (samples / 2) + 0.5
                samples = samples.detach().numpy().transpose(0, 2, 3, 1)
            fake_imgs.append(samples)
            real_imgs.append(data['img'].detach().numpy().transpose(0, 2, 3, 1))
            char_class.append(charclass)
            labels.append(condition)
        fake_imgs = np.concatenate(fake_imgs, axis=0)
        real_imgs = (np.concatenate(real_imgs, axis=0)/2) + 0.5
        char_class = torch.cat(char_class, dim=0)
        labels = torch.cat(labels, dim=0)
        return real_imgs, fake_imgs, labels, char_class
    elif gen =='multi':
        if mode == 'AC' or mode == 'CP':
            G_model = ACGenerator(weights, mask, z_dim=300, char_num=26, mode=mode, emb=opts.emb).to('cuda')
        elif mode == 'C':
            G_model = CGenerator(weights, mask, z_dim=300, char_num=26, mode=mode, emb=opts.emb).to('cuda')
        G_model = nn.DataParallel(G_model)
        G_model.load_state_dict(torch.load(model_path)["G_model_state_dict"], strict=False)
        G_model.eval()
        real_imgs = []
        fake_imgs = []
        char_class = []
        labels = []
        for iter, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            batch_size = len(data['img_target'])
            label, charclass = data['multi_embed_label_target'], data['charclass_target']
            noise = torch.normal(mean=0.5, std=0.2, size=(batch_size, 300))
            charclass = torch.eye(26)[charclass]
            condition = Multilabel_OneHot(label, len(ID), normalize=True)
            with torch.no_grad():
                samples = G_model(noise, condition, charclass)
                samples = samples.data.cpu()
                samples = (samples / 2) + 0.5
                samples = samples.detach().numpy().transpose(0, 2, 3, 1)
            fake_imgs.append(samples)
            real_imgs.append(data['img_target'].detach().numpy().transpose(0, 2, 3, 1))
            char_class.append(charclass)
            labels.append(condition)
        fake_imgs = np.concatenate(fake_imgs, axis=0)
        real_imgs = (np.concatenate(real_imgs, axis=0) / 2) + 0.5
        char_class = torch.cat(char_class, dim=0)
        labels = torch.cat(labels, dim=0)
        return real_imgs, fake_imgs, labels, char_class
def FID(real_imgs, fake_imgs):
    fid_score = FIDNumpy(batch_size=50, scaling=True)(real_imgs, fake_imgs)
    return fid_score
def gan_train_test(real_imgs, fake_imgs, label, mode = 'train', load = False):
    # real_imgs = np.repeat(real_imgs, 3, axis=3)
    # fake_imgs = np.repeat(fake_imgs, 3, axis=3)
    if gen == 'single':
        real_imgs = real_imgs.reshape(real_imgs.shape[0], real_imgs.shape[1], real_imgs.shape[2])
        fake_imgs = fake_imgs.reshape(fake_imgs.shape[0], fake_imgs.shape[1], fake_imgs.shape[2])
        model = models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(opts.char_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Sequential(
            nn.Linear(2048, len(ID))
        )
        model = model.to(device)
        model = nn.DataParallel(model)
        transform = transforms.Compose([
        transforms.ToTensor()])
        if mode == 'train':
            dataset = DataSet([fake_imgs, label], transform)
            test_dataset = DataSet([real_imgs, label], transform)
        elif mode == 'test':
            dataset = DataSet([real_imgs, label], transform)
            test_dataset = DataSet([fake_imgs, label], transform)
        #train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2)
        # train_dataset = torch.utils.data.Subset(dataset, train_indices)
        # val_dataset = torch.utils.data.Subset(dataset, val_indices)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        valloader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
        criterion = kl_divergence
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        print('===training start===')
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
        for epochs in range(20):
            running_loss = []
            running_acc = []
            validation_loss = []
            validation_acc = []
            model.train()
            for idx, (inputs, target) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
                inputs, target = inputs.to(device), target.to(device)
                optimizer.zero_grad()
                ops = model(inputs)
                train_loss = criterion(ops, target, activation='softmax')
                _, yy = torch.max(ops, 1)
                y_pred = yy.data.cpu().detach().numpy()
                target = target.data.cpu().detach().numpy()
                y_true = np.argmax(target, axis=1)
                acc =(np.sum(y_pred == y_true) / len(y_pred))
                running_loss.append(train_loss.item())
                running_acc.append(acc.item())
                train_loss.backward()  # 誤差逆伝播
                optimizer.step()  # パラメータ更新
            gc.collect()
            model.eval()
            for i, (xx, yy) in enumerate(valloader):
                xx, yy = xx.to(device), yy.to(device)
                pred = model(xx)
                loss = criterion(pred, yy, activation='softmax')
                _, y_pred = torch.max(pred, 1)
                y_pred = y_pred.data.cpu().detach().numpy()
                yy = yy.data.cpu().detach().numpy()
                y_true = np.argmax(yy, axis=1)
                pred_acc = (np.sum(y_pred == y_true) / len(y_pred))
                validation_loss.append(loss.item())
                validation_acc.append(pred_acc.item())
            total_batch_loss = np.asarray(running_loss).mean()
            total_val_loss = np.asarray(validation_loss).mean()
            total_batch_acc = np.asarray(running_acc).mean()
            total_val_acc = np.asarray(validation_acc).mean()
            print(f"Epoch: {epochs}, loss: {total_batch_loss}.")
            print(f"Epoch: {epochs}, loss: {total_val_loss}.")
            print(f"Epoch: {epochs}, accuracy: {total_batch_acc}.")
            print(f"Epoch: {epochs}, accuracy: {total_val_acc}.")
            history['train_loss'].append(total_batch_loss)
            history['val_loss'].append(total_val_loss)
            history['train_acc'].append(total_batch_acc)
            history['val_acc'].append(total_val_acc)
            torch.save(model.state_dict(), './experiment/real_model.path')
            print('===training finish===')
        model.eval()
        test_acc = []
        rare_test_acc = []
        key = list(ID.keys())
        values = [[]]* len(ID)
        word_score = dict(zip(key, values))
        yy_pred = []
        yy_true = []
        for i, (xx, yy) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            out = model(xx)
            _,  prediction= torch.max(out, 1)
            prediction = prediction.data.cpu().detach().numpy()
            yy = yy.data.cpu().detach().numpy()
            y_true = np.argmax(yy, axis=1)
            pred_acc = (np.sum(prediction == y_true) / len(prediction))
            test_acc.append(pred_acc)
            for i, key in enumerate(word_score.keys()):
                word_pred = prediction[y_true == i]
                word_score[key].extend(word_pred.tolist())
            prediction_ = list(prediction)
            y_true_= list(y_true)
            rare_score = np.array([aa == bb for aa, bb in zip(prediction_, y_true_) if bb in rare_index])
            rare_test_acc.append(rare_score.sum()/len(rare_score))
            yy_pred.append(out.data.cpu().detach().numpy())
            yy_true.append(y_true)
        word_score = {key:(np.array(values)==idx).sum()/len(values) for idx, (key, values) in enumerate(word_score.items())}
        total_test_acc = np.asarray(test_acc).mean()
        total_test_rare_acc = np.asarray(rare_test_acc).mean()
        history['test_acc'] = total_test_acc
        history['test_rare_acc'] = total_test_rare_acc
        history['word_score'] = word_score
        history['pred'] = np.concatenate(yy_pred, axis=0)
        history['y_true'] = np.concatenate(yy_true, axis=0)
        return history
    elif gen =='multi':
        real_imgs = real_imgs.reshape(real_imgs.shape[0], real_imgs.shape[1], real_imgs.shape[2])
        fake_imgs = fake_imgs.reshape(fake_imgs.shape[0], fake_imgs.shape[1], fake_imgs.shape[2])
        model = models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(opts.char_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Sequential(
            nn.Linear(2048, len(ID))
        )
        model = model.to(device)
        model = nn.DataParallel(model)
        transform = transforms.Compose([
            transforms.ToTensor()])
        if mode == 'train':
            dataset = DataSet([fake_imgs, label], transform)
            test_dataset = DataSet([real_imgs, label], transform)
        elif mode == 'test':
            dataset = DataSet([real_imgs, label], transform)
            test_dataset = DataSet([fake_imgs, label], transform)
        train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
        pos_weight = (len(label) - label.sum(0)) / label.sum(0)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        print('===training start===')
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
        for epochs in range(20):
            running_loss = []
            running_acc = []
            validation_loss = []
            validation_acc = []
            model.train()
            out_ = []
            yy_ = []
            for idx, (inputs, target) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
                inputs, target = inputs.to(device), target.to(device)
                target[target != 0] = 1.0
                optimizer.zero_grad()
                ops = model(inputs)
                out = torch.sigmoid(ops)
                #acc = mean_average_precision(out, target.data.cpu())
                train_loss = criterion(ops, target)
                running_loss.append(train_loss.item())
                out_.append(out.data.cpu())
                yy_.append(target.data.cpu())
                #running_acc.append(acc.item())
                train_loss.backward()  # 誤差逆伝播
                optimizer.step()  # パラメータ更新
            total_batch_acc = mean_average_precision(torch.cat(out_), torch.cat(yy_))
            model.eval()
            out_ = []
            yy_ = []
            gc.collect()
            for i, (xx, yy) in enumerate(valloader):
                xx, yy = xx.to(device), yy.to(device)
                yy[yy != 0] = 1.0
                ops = model(xx)
                out = torch.sigmoid(ops)
                #acc = mean_average_precision(out, yy.data.cpu())
                val_loss = criterion(ops, yy)
                validation_loss.append(val_loss.item())
                #validation_acc.append(acc.item())
                out_.append(out.data.cpu())
                yy_.append(yy.data.cpu())
            total_val_acc = mean_average_precision(torch.cat(out_), torch.cat(yy_))
            total_batch_loss = np.asarray(running_loss).mean()
            total_val_loss = np.asarray(validation_loss).mean()
            #total_batch_acc = np.asarray(running_acc).mean()
            #total_val_acc = np.asarray(validation_acc).mean()
            print(f"Epoch: {epochs}, loss: {total_batch_loss}.")
            print(f"Epoch: {epochs}, loss: {total_val_loss}.")
            print(f"Epoch: {epochs}, accuracy: {total_batch_acc}.")
            print(f"Epoch: {epochs}, accuracy: {total_val_acc}.")
            history['train_loss'].append(total_batch_loss)
            history['val_loss'].append(total_val_loss)
            history['train_acc'].append(total_batch_acc)
            history['val_acc'].append(total_val_acc)
        print('===training finish===')

        model.eval()
        test_acc = []
        rare_test_acc = []
        out_ = []
        yy_ = []
        for i, (xx, yy) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            yy[yy != 0] = 1.0
            out = model(xx)
            out = F.sigmoid(out)
            out_.append(out.data.cpu())
            yy_.append(yy.data.cpu())
        total_test_acc = mean_average_precision(torch.cat(out_), torch.cat(yy_))
        #total_test_acc = np.asarray(test_acc).mean()
        history['test_acc'] = total_test_acc
        return history
if __name__ == '__main__':
    GAN_type = 'Imp2Font'
    log_dir = os.path.join('./experiment', str(GAN_type), gen)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    real_imgs, fake_imgs, label, charclass = generator(Imp2Font_path, dataloader)
    label__ = torch.argmax(label, dim=1)
    # for idx, (img, ll) in tqdm.tqdm(enumerate(zip(real_imgs, label__)), total=len(real_imgs)):
    #     img = Image.fromarray((img*255).astype(np.uint8).reshape(img.shape[0], img.shape[1]))
    #     savedir = log_dir + "/" + str('real_image')
    #     os.makedirs(savedir, exist_ok=True)
    #     rare_savedir = log_dir + "/" + str('rare_real_image')
    #     os.makedirs(rare_savedir, exist_ok=True)
    #     savepath = savedir + "/" + str(idx).zfill(5) + ".png"
    #     img.save(savepath)
    #     if ll in rare_index:
    #         savepath = rare_savedir + "/" + str(idx).zfill(5) + ".png"
    #         img.save(savepath)
    # for idx, (img, ll) in tqdm.tqdm(enumerate(zip(fake_imgs, label__)), total=len(fake_imgs)):
    #     img = Image.fromarray((img * 255).astype(np.uint8).reshape(img.shape[0], img.shape[1]))
    #     savedir = log_dir + "/" + str('fake_image')
    #     os.makedirs(savedir, exist_ok=True)
    #     rare_savedir = log_dir + "/" + str('rare_fake_image')
    #     os.makedirs(rare_savedir, exist_ok=True)
    #     savepath = savedir + "/" + str(idx).zfill(5) + ".png"
    #     img.save(savepath)
    #     if ll in rare_index:
    #         savepath = rare_savedir + "/" + str(idx).zfill(5) + ".png"
    #         img.save(savepath)
    history = {}
    history1 = gan_train_test(real_imgs, fake_imgs, label, mode='train')
    history2 = gan_train_test(real_imgs, fake_imgs, label, mode='test')
    #fid_score = FID(real_imgs, fake_imgs)
    if gen == 'single':
        print('GAN-train:{}, GAN-test:{}, rare_GAN-train:{}, rare_GAN-test:{}'.format(history1['test_acc'], history2['test_acc'], history1['test_rare_acc'], history2['test_rare_acc']), )
    if gen == 'multi':
        print('GAN-train:{}, GAN-test:{}'.format(history1['test_acc'], history2['test_acc']))
    history["GAN-train"] = history1
    history["GAN-test"] = history2
    #history["FID"] = fid_score
    pickle_dump(history, os.path.join(log_dir, 'history'))