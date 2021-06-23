import torch
from torch import nn
from .common import *
from .self_attention import Self_Attn
from mylib import tile_like
class ACGenerator(nn.Module):
    def __init__(self,  weights, mask, z_dim = 300, num_dimension = 300, imp_num = 1574 , char_num = 26, mode = 'CP'):
        super(ACGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_dimension = num_dimension
        self.char_num = char_num
        self.imp_num = imp_num
        if mode == 'CP':
            sum_weight = True
            deepsets = False
        elif mode =='AC':
            sum_weight = False
            deepsets = False
        self.w2v_layer = ImpEmbedding(weights, mask, sum_weight= sum_weight, deepsets = deepsets)
        #self.Emb_layer = CustomEmbedding(num_dimension, 64, spectral_norm=True)
        self.layer1 = nn.Sequential(
            nn.Linear(self.z_dim + char_num , 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer2 = nn.Sequential(
            nn.Linear(self.num_dimension, 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer3 = nn.Sequential(
            nn.Linear(3000, 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # チャネル数を128⇒64に変える。
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True))
    # self.layer4 = Up(128, 64, num_dimension, 1, attention=True)

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                else:
                    continue
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                else:
                    continue
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                else:
                    continue
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                else:
                    continue

    def forward(self, noise, labels, char_class, w2v = True):
        y_1 = self.layer1(torch.cat([noise, char_class], dim=1))  # (100,1,1)⇒(300,1,1)
        # 印象情報のw2v
        if w2v:
            attr = self.w2v_layer(labels)
        else:
            attr = labels

        #印象情報のEmbedding
        # impression_id = torch.LongTensor(list(range(self.num_dimension)))
        # impression_id = impression_id.view(1, impression_id.size(0))
        # impression_id = impression_id.repeat(len(noise), 1)
        # impression_row = self.Emb_layer(impression_id.to(labels.device))
        # impression_feature = attr.unsqueeze(2) * impression_row

        y_2 = self.layer2(attr)  # (300,1,1)⇒(1500,1,1)
        x = torch.cat([y_1, y_2], 1)  # y_1 + y_2=(1800,1,1)
        x = self.layer3(x)  # (1800,1,1)⇒(512*8,1,1)
        x = x.view(-1, 128, 16, 16)  # (512,8,8)
        x = self.layer4(x)  # (512,8,8)⇒(256,16,16)
        x = self.layer5(x)  # (256,16,16)⇒(128,32,32)
        return x

class ACDiscriminator(nn.Module):
    def __init__(self, weight, mask, img_size = 64,  num_dimension = 300, imp_num = 1574, char_num = 26 ,mode = 'CP'):
        super(ACDiscriminator, self).__init__()
        self.num_dimension = num_dimension
        self.imp_num = imp_num
        self.img_size = img_size
        self.char_num = char_num
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1 + char_num, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
            )


        self.fc_TF = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128 * 16 * 16, 1024)),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )
        self.fc_class =nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128 * 16 * 16, 1024)),
            # nn.BatchNorm1d(1024),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.imp_num)
        )
        self.fc_char = nn.Linear(128 * 16 * 16, char_num)

        self.init_weights()


    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, img, label, char_class):
        char = char_class.view(char_class.size(0), char_class.size(1), 1, 1).expand(-1, -1, self.img_size, self.img_size)
        x = self.layer1(torch.cat([img, char], dim=1))
        x = self.layer2(x)
        x = x.view(-1, 128 * 16 * 16)
        x_TF = self.fc_TF(x)
        x_class = self.fc_class(x)
        return x_TF, x_class

class CGenerator(nn.Module):
    def __init__(self,  weights, mask, z_dim = 300, num_dimension = 300, imp_num = 1574 , char_num = 26, attention = True):
        super(CGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_dimension = num_dimension
        self.char_num = char_num
        self.imp_num = imp_num
        self.w2v_layer = ImpEmbedding(weights, mask, sum_weight=False, deepsets=False)
        #self.Emb_layer = CustomEmbedding(num_dimension, 64, spectral_norm=True)
        self.layer1 = nn.Sequential(
            nn.Linear(self.z_dim + char_num , 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer2 = nn.Sequential(
            nn.Linear(self.num_dimension, 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer3 = nn.Sequential(
            nn.Linear(3000, 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # チャネル数を128⇒64に変える。
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True))
    # self.layer4 = Up(128, 64, num_dimension, 1, attention=True)

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, noise, labels, char_class, w2v = True):
        y_1 = self.layer1(torch.cat([noise, char_class], dim=1))  # (100,1,1)⇒(300,1,1)
        # 印象情報のw2v
        if w2v:
            attr = self.w2v_layer(labels)
        else:
            attr = labels
        #印象情報のEmbedding
        # impression_id = torch.LongTensor(list(range(self.num_dimension)))
        # impression_id = impression_id.view(1, impression_id.size(0))
        # impression_id = impression_id.repeat(len(noise), 1)
        # impression_row = self.Emb_layer(impression_id.to(labels.device))
        # impression_feature = attr.unsqueeze(2) * impression_row

        y_2 = self.layer2(attr)  # (300,1,1)⇒(1500,1,1)
        x = torch.cat([y_1, y_2], 1)  # y_1 + y_2=(1800,1,1)
        x = self.layer3(x)  # (1800,1,1)⇒(512*8,1,1)
        x = x.view(-1, 128, 16, 16)  # (512,8,8)
        x = self.layer4(x)  # (512,8,8)⇒(256,16,16)
        x = self.layer5(x)  # (256,16,16)⇒(128,32,32)
        return x

class CDiscriminator(nn.Module):
    def __init__(self, weight, mask, img_size=64, num_dimension=300, imp_num=1574, char_num=26):
        super(CDiscriminator, self).__init__()
        self.num_dimension = num_dimension
        self.imp_num = imp_num
        self.img_size = img_size
        self.char_num = char_num
        self.w2v_layer = ImpEmbedding(weight, mask, sum_weight=False)
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1 + char_num +num_dimension, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc_TF = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128 * 16 * 16, 1024)),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, img, label, char_class):
        char = char_class.view(char_class.size(0), char_class.size(1), 1, 1).expand(-1, -1, self.img_size, self.img_size)
        attr = self.w2v_layer(label)
        attr = attr.view(attr.size(0), attr.size(1), 1, 1).expand(-1, -1, self.img_size, self.img_size)
        x = self.layer1(torch.cat([img, char, attr], dim=1))
        x = self.layer2(x)
        x = x.view(-1, 128 * 16 * 16)
        x_TF = self.fc_TF(x)
        return x_TF