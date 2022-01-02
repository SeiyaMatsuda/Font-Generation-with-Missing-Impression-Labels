from .common import *
class Imp2font_ImpEmbedding(nn.Module):
    def __init__(self, weights, sum_weight = True, deepsets = False,  num_dimension = 300, residual_num = 0, required_grad = False):
        super(Imp2font_ImpEmbedding, self).__init__()
        self.weight = weights
        self.embed = nn.Embedding(self.weight.shape[0], self.weight.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(self.weight))
        self.shape = (self.weight.shape[0], self.weight.shape[1])
        self.sum_weight = sum_weight
        self.deepsets = deepsets
        print('deepsets:{}::::sum_weight:{}'.format(self.deepsets, self.sum_weight))
        if sum_weight:
            self.fc =nn.Sequential(
                 nn.Linear(self.weight.shape[0], 1),
                 nn.LeakyReLU(0.2, inplace=True))
        if deepsets:
            self.sets_layer = DeepSets(num_dimension, num_dimension)
        if not required_grad:
            self.embed.weight.requires_grad = False
        res_block =[]
        for i in range(residual_num):
            res_block.append(ResidualBlock(num_dimension))
        self.res_block = nn.Sequential(*res_block)
    def forward(self, labels , w2v = True):
        if w2v:
            labels = labels.view(labels.size()[0], labels.size()[1], 1)
            attr = torch.mul(self.embed.weight.data, labels)
        else:
            attr = labels
        if self.sum_weight:
            attr = self.fc(attr.permute(0, 2, 1))
            attr = attr.permute(0, 2, 1)
            attr = attr.sum(1)
        elif self.deepsets:
            attr = self.sets_layer(attr)
        else:
            attr = attr.sum(1)
        attr = self.res_block(attr)
        return attr

class Imp2Font(nn.Module):
    def __init__(self,  weights, latent_size=300, w2v_dimension=300, imp_num=1574, char_num=26, sum_weight=True, deepsets=True):
        super(Imp2Font, self).__init__()
        self.z_dim = latent_size
        self.char_num = char_num
        self.imp_num = imp_num
        self.w2v_layer = Imp2font_ImpEmbedding(weights, sum_weight=sum_weight, deepsets=deepsets)
        self.num_dimension = w2v_dimension
        self.layer1 = nn.Sequential(
            nn.Linear(self.z_dim + char_num , 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer2 = nn.Sequential(
            nn.Linear(self.num_dimension, 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer3 = nn.Sequential(
            nn.Linear(3000 , 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # チャネル数を128⇒64に変える。
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True))

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

    def forward(self, noise, y_char, y_imp, res):
        y_1 = self.layer1(torch.cat([noise, y_char], dim=1))  # (100,1,1)⇒(300,1,1)
        # 印象情報のw2v
        attr = self.w2v_layer(y_imp)
        #印象情報のEmbedding
        y_2 = self.layer2(attr)  # (300,1,1)⇒(1500,1,1)
        x = torch.cat([y_1, y_2], dim = 1)  # y_1 + y_2=(1800,1,1)
        x = self.layer3(x)  # (1800,1,1)⇒(512*8,1,1)
        x = x.view(-1, 128, 16, 16)  # (512,8,8)
        x = self.layer4(x)  # (512,8,8)⇒(256,16,16)
        x = self.layer5(x)  # (256,16,16)⇒(128,32,32)
        return x, attr


class ACGenerator(nn.Module):
    def __init__(self,  weights, z_dim = 300, num_dimension = 300, imp_num = 1574 , char_num = 26, mode = 'CP', emb = 'w2v'):
        super(ACGenerator, self).__init__()
        self.z_dim = z_dim
        self.char_num = char_num
        self.imp_num = imp_num
        self.emb = emb
        if mode == 'CP':
            sum_weight = True
            deepsets = False
        elif mode =='AC':
            sum_weight = False
            deepsets = False
        if emb == 'w2v':
            self.w2v_layer = ImpEmbedding(weights, sum_weight= sum_weight, deepsets = deepsets)
        elif emb == 'one-hot':
            num_dimension = imp_num
        self.num_dimension = num_dimension
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
            nn.Linear(3000 , 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # チャネル数を128⇒64に変える。
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True))
        #self.layer4 = Up(128, 64, num_dimension, 1, attention=True)

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
        if self.emb=='w2v':
            if w2v:
                attr = self.w2v_layer(labels)
            else:
                attr = self.w2v_layer(labels, w2v = w2v)
        elif self.emb=='one-hot':
            attr = labels
        #印象情報のEmbedding
        # impression_id = torch.LongTensor(list(range(self.num_dimension)))
        # impression_id = impression_id.view(1, impression_id.size(0))
        # impression_id = impression_id.repeat(len(noise), 1)
        # impression_row = self.Emb_layer(impression_id.to(labels.device))
        # impression_feature = attr.unsqueeze(2) * impression_row
        y_2 = self.layer2(attr)  # (300,1,1)⇒(1500,1,1)
        x = torch.cat([y_1, y_2], dim = 1)  # y_1 + y_2=(1800,1,1)
        x = self.layer3(x)  # (1800,1,1)⇒(512*8,1,1)
        x = x.view(-1, 128, 16, 16)  # (512,8,8)
        x = self.layer4(x)  # (512,8,8)⇒(256,16,16)
        x = self.layer5(x)  # (256,16,16)⇒(128,32,32)
        return x
