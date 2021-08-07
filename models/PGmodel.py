from .common import *
from torchvision import models
import numpy as np
import torch.nn.functional as F


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
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(outch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),

            ]

        else:
            layers = [
                nn.Upsample((out_size, out_size), mode='nearest'),
                Conv2d(inch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(outch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
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

    def __init__(self, out_size, inch, outch, char_num=26, imp_num=1574, final=False):
        super().__init__()
        self.final = final
        if final:
            layers = [
                Minibatch_std(),  # final block only
                Conv2d(inch + 1, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Conv2d(outch, outch, 4, padding=0),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            layer_TF = [nn.Conv2d(outch, 1, 1, padding=0)]
            layer_char = [nn.Conv2d(outch, char_num, 1, padding=0)]
            layer_imp = [nn.Flatten(),
                         nn.Dropout(p=0.5),
                         nn.Linear(outch * 4 * 4, imp_num),]

            self.layer_TF = nn.Sequential(*layer_TF)
            self.layer_char = nn.Sequential(*layer_char)
            self.layer_imp = nn.Sequential(*layer_imp)
        else:
            layers = [
                Conv2d(inch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                Conv2d(outch, outch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.AdaptiveAvgPool2d((out_size, out_size)),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x_ = self.layers(x)
        if self.final:
            x_TF = torch.squeeze(self.layer_TF(x_))
            x_char = torch.squeeze(self.layer_char(x_))
            x_imp = self.layers[:-2](x)
            x_imp = torch.squeeze(self.layer_imp(x_imp))
        else:
            x_TF = x_
            x_char = None
            x_imp = None
        return x_TF, x_char, x_imp


#
class Generator(nn.Module):
    def __init__(self, weight, latent_size=256, char_num=26, num_dimension=300, attention=False, device=torch.device("cuda")):
        super().__init__()

        # conv modules & toRGBs
        self.attention = attention
        scale = 1
        inchs = np.array([latent_size + char_num + num_dimension, 256, 128, 64, 32, 16], dtype=np.uint32) * scale
        outchs = np.array([256, 128, 64, 32, 16, 8], dtype=np.uint32) * scale
        sizes = np.array([4, 8, 16, 32, 64, 128], dtype=np.uint32)
        firsts = np.array([True, False, False, False, False, False], dtype=np.bool)
        blocks, toRGBs, attn_blocks = [], [], []
        for idx, (s, inch, outch, first) in enumerate(zip(sizes, inchs, outchs, firsts)):
            blocks.append(ConvModuleG(s, inch, outch, first))
            toRGBs.append(nn.Conv2d(outch, 1, 1, padding=0))
            if attention:
                attn_blocks.append(Attention(outch, num_dimension, len(sizes) - (idx + 1)))
        self.emb_layer = ImpEmbedding(weight, deepsets=False, device=device)
        self.CA_layer = Conditioning_Augumentation(num_dimension + char_num, latent_size, device=device)
        self.blocks = nn.ModuleList(blocks)
        self.toRGBs = nn.ModuleList(toRGBs)
        if attention:
            self.attn_blocks = nn.ModuleList(attn_blocks)
            self.attribute_embed = nn.Embedding(num_dimension, 128)
            attrid = torch.tensor([i for i in range(num_dimension)])
            self.attrid = attrid.view(1, attrid.size(0))

        self.size = sizes

    def forward(self, x, y_char, y_imp, res, eps=1e-7, emb=True):
        # to image
        n, c = x.shape
        x = x.reshape(n, c // 16, 4, 4)
        if emb:
            y_sc = self.emb_layer(y_imp)
        y_cond = torch.cat([y_sc, y_char], dim=1)
        y_cond = y_cond.reshape(y_cond.size(0), y_cond.size(1), 1, 1)
        y_cond = y_cond.expand(y_cond.size(0), y_cond.size(1), 4, 4)
        # attribute embedding
        if self.attention:
            attrid = self.attrid.repeat(x.size(0), 1).to(y_imp.device)
            attr_raw = self.attribute_embed(attrid)
            y_emb = y_imp.unsqueeze(2) * attr_raw
        # for the highest resolution
        x = torch.cat([x, y_cond], axis=1)
        res = min(res, len(self.blocks))

        # get integer by floor
        nlayer = max(int(res - eps), 0)
        for i in range(nlayer):
            x = self.blocks[i](x)
            if self.attention:
                x = self.attn_blocks[i](x, y_emb)
        # high resolution

        x_big = self.blocks[nlayer](x)
        if self.attention:
            x_big = self.attn_blocks[nlayer](x_big, y_emb)
        dst_big = self.toRGBs[nlayer](x_big)

        if nlayer == 0:
            x = dst_big
        else:
            # low resolution
            x_sml = F.interpolate(x, x_big.shape[2:4], mode='nearest')
            dst_sml = self.toRGBs[nlayer - 1](x_sml)
            alpha = res - int(res - eps)
            x = (1 - alpha) * dst_sml + alpha * dst_big

        mu, logvar = None, None
        return torch.tanh(x), mu, logvar


class Discriminator(nn.Module):
    def __init__(self, imp_num=1574, char_num=26, device=torch.device("cuda")):
        super().__init__()

        self.minbatch_std = Minibatch_std()

        # conv modules & toRGBs
        scale = 1
        inchs = np.array([256, 128, 64, 32, 16, 8], dtype=np.uint32) * scale
        outchs = np.array([512, 256, 128, 64, 32, 16], dtype=np.uint32) * scale
        sizes = np.array([1, 4, 8, 16, 32, 64], dtype=np.uint32)
        finals = np.array([True, False, False, False, False, False], dtype=np.bool)
        blocks, fromRGBs = [], []
        for s, inch, outch, final in zip(sizes, inchs, outchs, finals):
            fromRGBs.append(nn.Conv2d(1, inch, 1, padding=0))
            blocks.append(ConvModuleD(s, inch, outch, imp_num=imp_num, char_num=char_num, final=final))

        self.fromRGBs = nn.ModuleList(fromRGBs)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, res):
        # for the highest resolution
        res = min(res, len(self.blocks))

        # get integer by floor
        eps = 1e-7
        n = max(int(res - eps), 0)

        # high resolution
        x_big = self.fromRGBs[n](x)
        x_big, char, imp = self.blocks[n](x_big)

        if n == 0:
            x = x_big
        else:
            # low resolution
            x_sml = F.adaptive_avg_pool2d(x, x_big.shape[2:4])
            x_sml = self.fromRGBs[n - 1](x_sml)
            alpha = res - int(res - eps)
            x = (1 - alpha) * x_sml + alpha * x_big

        for i in range(n):
            x, char, imp = self.blocks[n - 1 - i](x)
        return x, char, imp
