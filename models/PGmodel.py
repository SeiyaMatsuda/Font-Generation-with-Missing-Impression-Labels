from .common import *
from pytorch_revgrad import RevGrad
from torchvision import models

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

    def __init__(self, out_size, inch, outch, char_num = 26, imp_num = 1574, final=False):
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
            # layer_imp = [nn.Conv2d(outch, imp_num, 1, padding=0)]
            layer_imp = [nn.Flatten(),
                nn.Linear(outch * 4 * 4, 1024),
                nn.Dropout(p=0.5),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, imp_num)]

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
# class ConvModuleUnet(nn.Module):
#     '''
#     Args:
#         out_size: (int), Ex.: 16 (resolution)
#         inch: (int),  Ex.: 256
#         outch: (int), Ex.: 128
#     '''
#     def __init__(self, out_size, inch, outch):
#         super().__init__()
#
#         layers = [
#             nn.Upsample((out_size, out_size), mode='nearest'),
#             Conv2d(inch, outch, 3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             Conv2d(outch, outch, 3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#         ]
#
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.layers(x)
# class DeConvModuleUnet(nn.Module):
#     '''
#     Args:
#         out_size: (int), Ex.: 16 (resolution)
#         inch: (int),  Ex.: 256
#         outch: (int), Ex.: 128
#     '''
#
#     def __init__(self, out_size, inch, outch, char_num = 26, imp_num = 1574, final=False):
#         super().__init__()
#         layers = [
#             Conv2d(inch, outch, 3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             Conv2d(outch, outch, 3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.AdaptiveAvgPool2d((out_size, out_size)),
#         ]
#
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.layers(x)
#         return x

class Generator(nn.Module):
    def __init__(self, weight, latent_size=512, char_num=26, attention=False):
        super().__init__()

        # conv modules & toRGBs
        self.attention = attention
        scale = 1
        inchs  = np.array([latent_size + char_num + weight.shape[1], 256, 128,64,32,16], dtype=np.uint32)*scale
        outchs = np.array([256, 128, 64, 32, 16, 8], dtype=np.uint32)*scale
        sizes = np.array([4, 8, 16, 32, 64, 128], dtype=np.uint32)
        firsts = np.array([True, False, False, False, False, False],  dtype=np.bool)
        blocks, toRGBs, attn_blocks = [], [], []
        for idx, (s, inch, outch, first) in enumerate(zip(sizes, inchs, outchs, firsts)):
            blocks.append(ConvModuleG(s, inch, outch, first))
            toRGBs.append(nn.Conv2d(outch, 1, 1, padding=0))
            if attention:
                attn_blocks.append(Attention(outch, weight.shape[1], len(sizes) - (idx+1)))
        self.emb_layer = ImpEmbedding(weight, sum_weight=False, deepsets=True)
        self.blocks = nn.ModuleList(blocks)
        self.toRGBs = nn.ModuleList(toRGBs)
        if attention:
            self.attn_blocks = nn.ModuleList(attn_blocks)
            self.attribute_embed = nn.Embedding(weight.shape[1], 128)
            attrid = torch.tensor([i for i in range(weight.shape[1])])
            self.attrid = attrid.view(1, attrid.size(0))

        self.size = sizes
        self.RevGrad = RevGrad()

    def forward(self, x, y_char, y_imp, res, eps=1e-7, RevGrad = False):
        # to image
        n, c = x.shape
        x = x.reshape(n, c//16, 4, 4)
        # char vector
        y_char = y_char.reshape(y_char.size(0), y_char.size(1), 1, 1)
        y_char = y_char.expand(y_char.size(0), y_char.size(1),4,4)
        # impression embedding
        y_imp = self.emb_layer(y_imp)
        y_sc = y_imp.reshape(y_imp.size(0), y_imp.size(1), 1, 1)
        y_sc = y_sc.expand(y_sc.size(0), y_sc.size(1),4,4)
        # attribute embedding
        if self.attention:
            attrid = self.attrid.repeat(x.size(0), 1).to(y_imp.device)
            attr_raw = self.attribute_embed(attrid)
            y_emb = y_imp.unsqueeze(2) * attr_raw
        # for the highest resolution
        x = torch.cat([x, y_char, y_sc], axis = 1)
        res = min(res, len(self.blocks))

        # get integer by floor
        nlayer = max(int(res-eps), 0)
        for i in range(nlayer):
            x = self.blocks[i](x)
            if self.attention:
                x = self.attn_blocks[i](x, y_emb)
        # high resolution

        x_big = self.blocks[nlayer](x)
        if self.attention:
            x_big = self.attn_blocks[nlayer](x_big, y_emb)
        dst_big = self.toRGBs[nlayer](x_big)

        if nlayer==0:
            x = dst_big
        else:
            # low resolution
            x_sml = F.interpolate(x, x_big.shape[2:4], mode='nearest')
            dst_sml = self.toRGBs[nlayer-1](x_sml)
            alpha = res - int(res-eps)
            x = (1-alpha)*dst_sml + alpha*dst_big

        if RevGrad == True:
            x = self.RevGrad(x)
        return torch.sigmoid(x), y_imp

class Discriminator(nn.Module):
    def __init__(self, imp_num = 1574, char_num = 26):
        super().__init__()

        self.minbatch_std = Minibatch_std()

        # conv modules & toRGBs
        scale = 1
        inchs = np.array([256,128, 64,32,16, 8], dtype=np.uint32)*scale
        outchs  = np.array([512,256,128,64,32,16], dtype=np.uint32)*scale
        sizes = np.array([1,4,8,16,32,64], dtype=np.uint32)
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
        n = max(int(res-eps), 0)

        # high resolution
        x_big = self.fromRGBs[n](x)
        x_big, char, imp = self.blocks[n](x_big)

        if n==0:
            x = x_big
        else:
            # low resolution
            x_sml = F.adaptive_avg_pool2d(x, x_big.shape[2:4])
            x_sml = self.fromRGBs[n-1](x_sml)
            alpha = res - int(res-eps)
            x = (1-alpha)*x_sml + alpha*x_big

        for i in range(n):
            x, char, imp = self.blocks[n-1-i](x)
        return x, char, imp
#