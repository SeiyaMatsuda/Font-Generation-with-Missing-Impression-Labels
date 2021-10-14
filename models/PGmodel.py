from .common import *
class Mapping_net(nn.Module):
    def __init__(self, dim_latent, n_fc):
        super().__init__()
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_fc):
            layers.append(nn.Linear(dim_latent, dim_latent))
            if i == n_fc-1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)

class ConvModuleG(nn.Module):
    '''
    Args:
        out_size: (int), Ex.: 16 (resolution)
        inch: (int),  Ex.: 256
        outch: (int), Ex.: 128
    '''
    def __init__(self, out_size, inch, outch, first=False):
        super().__init__()

        if first:
            layers = [
                        Conv2d(inch, outch, 3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        Conv2d(outch, outch, 3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),

            ]

        else:
            layers = [
                nn.Upsample((out_size, out_size), mode='nearest'),
                Conv2d(inch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv2d(outch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvModuleD(nn.Module):
    '''
    Args:
        out_size: (int), Ex.: 16 (resolution)
        inch: (int),  Ex.: 256
        outch: (int), Ex.: 128
    '''

    def __init__(self, out_size, inch, outch, num_dimension=300, char_num=26, imp_num = 1574, final=False, dropout=False):
        super().__init__()
        self.final = final
        if final:
            layers = [
                Minibatch_std(),  # final block only
                Conv2d(inch + 1, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv2d(outch + num_dimension, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if dropout==True:
                layers.insert(4, nn.Dropout2d(0.5))
            layer_TF = [nn.Conv2d(outch, 1, 4, padding=0)]
            layer_char = [nn.Conv2d(outch, char_num, 4, padding=0)]
            layer_imp = [
                nn.Flatten(),
                nn.Linear(outch * 4 * 4, 300),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(300, imp_num)]

            self.layer_TF = nn.Sequential(*layer_TF)
            self.layer_char = nn.Sequential(*layer_char)
            self.layer_imp = nn.Sequential(*layer_imp)
        else:
            layers = [
                Conv2d(inch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv2d(outch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d((out_size, out_size)),
            ]
            if dropout==True:
                layers.insert(3, nn.Dropout2d(0.5))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, cond=None):
        if self.final:
            if cond==None:
                x = self.layers[:3](x)
                x_TF, x_char = None, None
                x_imp = torch.squeeze(self.layer_imp(x))
            else:
                x_ = self.layers[:3](x)
                x_imp = torch.squeeze(self.layer_imp(x_))
                cond_ = cond.view(cond.size(0), cond.size(1), 1, 1).repeat(1, 1, x_.size(2),  x_.size(3))
                x_ = torch.cat((x_, cond_), axis=1)
                x_ = self.layers[3:](x_)
                x_TF = torch.squeeze(self.layer_TF(x_))
                x_char = torch.squeeze(self.layer_char(x_))
        else:
            x_ = self.layers(x)
            x_TF = x_
            x_char = None
            x_imp = None
        return x_TF, x_char, x_imp

class Generator(nn.Module):
    def __init__(self, weight, latent_size=512, w2v_dimension=300, num_dimension=300, char_num=26, attention=False):
        super().__init__()

        # conv modules & toRGBs
        self.attention = attention
        self.weight = torch.tensor(weight)
        scale = 1
        inchs  = np.array([latent_size + char_num + num_dimension, 256, 128, 64, 32], dtype=np.uint32)*scale
        outchs = np.array([256, 128, 64, 32, 16], dtype=np.uint32)*scale
        sizes = np.array([4, 8, 16, 32, 64], dtype=np.uint32)
        firsts = np.array([True, False, False, False, False],  dtype=np.bool)
        blocks, toRGBs, attn_blocks = [], [], []
        for idx, (s, inch, outch, first) in enumerate(zip(sizes, inchs, outchs, firsts)):
            blocks.append(ConvModuleG(s, inch, outch, first))
            toRGBs.append(nn.Conv2d(outch, 1, 1, padding=0))
            if attention:
                attn_blocks.append(DCAN(outch, num_dimension, s))
        # self.mapping = Mapping_net(dim_latent=weight.shape[0], n_fc=8)
        self.emb_layer = ImpEmbedding(weight, sum_weight=False, deepsets=False)
        self.CA_layer = Conditioning_Augumentation(w2v_dimension, num_dimension)
        self.blocks = nn.ModuleList(blocks)
        self.toRGBs = nn.ModuleList(toRGBs)
        if attention:
            self.attn_blocks = nn.ModuleList(attn_blocks)
        self.size = sizes
    def impression_embedding(self, y_imp):
        y_imp = self.emb_layer(y_imp)
        return y_imp
    def mean_embedding_representation(self, y_imp):
        y_imp = torch.mul(self.weight/(torch.linalg.norm(self.weight, dim=1).unsqueeze(1) + 1e-7), y_imp.unsqueeze(2)).mean(axis=1)
        return y_imp
    def forward(self, z, y_char, y_imp, res, eps=1e-7,  emb=True):
        # to image
        x = z[0]
        z_cond = z[1]
        n, c = x.shape
        x = x.reshape(n, c//16, 4, 4)
        # char vector
        y_char = y_char.reshape(y_char.size(0), y_char.size(1), 1, 1)
        y_char = y_char.expand(y_char.size(0), y_char.size(1),4,4)
        # impression embedding
        if emb:
            y_imp = self.emb_layer(y_imp)
        cond, mu, logvar = self.CA_layer(y_imp, z_cond)
        y_sc = cond.reshape(cond.size(0), cond.size(1), 1, 1)
        y_sc = y_sc.expand(y_sc.size(0), y_sc.size(1), 4, 4)
        # for the highest resolution
        x = torch.cat([x, y_char, y_sc], axis = 1)
        res = min(res, len(self.blocks))
        # get integer by floor
        nlayer = max(int(res-eps), 0)
        for i in range(nlayer):
            x = self.blocks[i](x)
            if self.attention:
                x, mask = self.attn_blocks[i](x, cond)
        # high resolution

        x_big = self.blocks[nlayer](x)
        if self.attention:
            x_big, mask = self.attn_blocks[nlayer](x_big, cond)
        dst_big = self.toRGBs[nlayer](x_big)

        if nlayer==0:
            x = dst_big
        else:
            # low resolution
            x_sml = F.interpolate(x, x_big.shape[2:4], mode='nearest')
            dst_sml = self.toRGBs[nlayer-1](x_sml)
            alpha = res - int(res-eps)
            x = (1-alpha)*dst_sml + alpha*dst_big
        return torch.tanh(x), mu, logvar

class Discriminator(nn.Module):
    def __init__(self, num_dimension=300, imp_num=1574, char_num=26):
        super().__init__()

        self.minbatch_std = Minibatch_std()

        # conv modules & toRGBs
        scale = 1
        inchs = np.array([256, 128, 64, 32, 16], dtype=np.uint32)*scale
        outchs  = np.array([512, 256, 128, 64, 32], dtype=np.uint32)*scale
        sizes = np.array([1, 4, 8, 16, 32], dtype=np.uint32)
        finals = np.array([True, False, False, False, False], dtype=np.bool)
        dropouts = np.array([True, True, True, False, False], dtype=np.bool)
        blocks, fromRGBs = [], []
        for s, inch, outch, final, dropout in zip(sizes, inchs, outchs, finals, dropouts):
            fromRGBs.append(nn.Conv2d(1, inch, 1, padding=0))
            blocks.append(ConvModuleD(s, inch, outch, num_dimension=num_dimension, imp_num=imp_num, char_num=char_num, final=final, dropout=dropout))

        self.fromRGBs = nn.ModuleList(fromRGBs)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, res, cond=None):
        # for the highest resolution
        res = min(res, len(self.blocks))

        # get integer by floor
        eps = 1e-7
        n = max(int(res-eps), 0)

        # high resolution
        x_big = self.fromRGBs[n](x)
        x_big, char, imp = self.blocks[n](x_big, cond)

        if n==0:
            x = x_big
        else:
            # low resolution
            x_sml = F.adaptive_avg_pool2d(x, x_big.shape[2:4])
            x_sml = self.fromRGBs[n-1](x_sml)
            alpha = res - int(res-eps)
            x = (1-alpha)*x_sml + alpha*x_big

        for i in range(n):
            x, char, imp = self.blocks[n-1-i](x, cond)
        return x, char, imp
#
#