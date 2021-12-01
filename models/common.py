import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from models.Deepsets import DeepSets
from models.DeformConv2d import DeformConv2d

class CustomConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 bias=True,
                 spectral_norm=False,
                 residual_init=True):
        super(CustomConv2d, self).__init__()
        self.residual_init = residual_init
        if padding is None:
            padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        if spectral_norm:
            self.conv = utils.spectral_norm(self.conv)

    def forward(self, input):
        return self.conv(input)

class CustomLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 spectral_norm=False):
        super(CustomLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if spectral_norm:
            self.linear = utils.spectral_norm(self.linear)

    def forward(self, input):
        return self.linear(input)


class ConvMeanPool(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 spectral_norm=False,
                 residual_init=True):
        super(ConvMeanPool, self).__init__()
        self.conv = CustomConv2d(in_channels,
                                 out_channels,
                                 kernel_size,
                                 bias=bias,
                                 spectral_norm=spectral_norm,
                                 residual_init=residual_init)

    def forward(self, input):
        output = input
        output = self.conv(output)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        return output


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 resample=None,
                 spectral_norm=False):
        super(ResidualBlock, self).__init__()
        if in_channels != out_channels or resample is not None:
            self.learnable_shortcut = True
        else:
            self.learnable_shortcut = False

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.conv_shortcut = ConvMeanPool(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              spectral_norm=spectral_norm,
                                              residual_init=False)
            self.conv1 = CustomConv2d(in_channels,
                                      in_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
            self.conv2 = ConvMeanPool(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
        elif resample is None:
            if self.learnable_shortcut:
                self.conv_shortcut = CustomConv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=1,
                                                  spectral_norm=spectral_norm,
                                                  residual_init=False)
            self.conv1 = CustomConv2d(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
            self.conv2 = CustomConv2d(out_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
        else:
            raise NotImplementedError()

    def forward(self, input):
        if self.learnable_shortcut:
            shortcut = self.conv_shortcut(input)
        else:
            shortcut = input

        output = input
        output = self.relu1(output)
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output

    def forward(self, input):
        shortcut = self.conv_shortcut(input)

        output = input
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output

class ImpEmbedding(nn.Module):
    def __init__(self, weight,  num_dimension=300, residual_num=0,  deepsets=False, normalize=True, required_grad=False):
        super(ImpEmbedding, self).__init__()
        self.weight = weight
        self.embed = nn.Embedding(self.weight.shape[0], self.weight.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(self.weight))
        self.shape = (self.weight.shape[0], self.weight.shape[1])
        self.deepsets = deepsets
        self.normalize = normalize
        print('deepsets:{}'.format(self.deepsets))
        if deepsets:
            self.sets_layer = DeepSets(num_dimension, num_dimension)
        if not required_grad:
            self.embed.weight.requires_grad = False
        res_block =[]
        for i in range(residual_num):
            res_block.append(ResidualBlock(num_dimension))
        self.res_block = nn.Sequential(*res_block)
    def forward(self, labels , w2v=True):
        labels = labels.view(labels.size(0), labels.size(1), 1)
        attr = torch.mul(self.embed.weight.data, labels)
        if self.deepsets:
            attr = self.sets_layer(attr)
        else:
            attr = attr.sum(1)
            if self.normalize:
                attr = attr/(torch.sqrt((attr ** 2).sum(1)).unsqueeze(1) + 1e-7)
        return attr
class Conditioning_Augumentation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Conditioning_Augumentation, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.output_dim]
        logvar = x[:, self.output_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar, eps):
        std = logvar.mul(0.5).exp_()
        eps = Variable(eps)
        return mu + eps.mul(std)

    def forward(self, text_embedding, eps):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar, eps)
        return c_code, mu, logvar


class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Linear(in_channel, in_channel, bias=False),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel, in_channel,bias=False),
            nn.BatchNorm1d(in_channel),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class CALayer2d(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer2d, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # print('attention size', x.size())
        m_batchsize, C, width, height = x.size()
        # print('query_conv size', self.query_conv(x).size())
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C X (W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out
class Attention(nn.Module):
    def __init__(self, in_channel, attr_channel, attr_down_scale):
        super(Attention, self).__init__()
        attr_layers = []
        for _ in range(attr_down_scale):
            attr_layers += [nn.Conv2d(attr_channel, attr_channel, kernel_size=4,
                                      stride=2, padding=1, bias=True),
                            CALayer2d(attr_channel),
                            nn.LeakyReLU(0.2, inplace=True)]
        self.attr_layer = nn.Sequential(*attr_layers)

        img_attrs = []
        img_attrs += [nn.Conv2d(attr_channel + in_channel, in_channel, 3,
                                stride=1, padding=1, bias=False),
                      nn.InstanceNorm2d(in_channel),
                      nn.LeakyReLU(0.2, inplace=True)]
        img_attrs += [CALayer2d(in_channel),
                      CALayer2d(in_channel),
                      SelfAttention(in_channel),
                      nn.LeakyReLU(0.2, inplace=True)]
        self.img_attr = nn.Sequential(*img_attrs)

    def forward(self, x, attr_feature):
        attr_feature = attr_feature.unsqueeze(-1)
        attr_featureT = torch.transpose(attr_feature, 2, 3)
        attr_feature2d = torch.matmul(attr_feature, attr_featureT)
        attr = self.attr_layer(attr_feature2d)
        out = torch.cat([x, attr], 1)
        out = self.img_attr(out)
        return out

class DCAN(nn.Module):
    def __init__(self, in_channel, attr_channel, attr_down_scale):
        super().__init__()
        attr_layers = []
        for _ in range(attr_down_scale):
            attr_layers += [nn.Conv2d(attr_channel, attr_channel, kernel_size=4,
                                      stride=2, padding=1, bias=True),
                            CALayer2d(attr_channel),
                            nn.LeakyReLU(0.2, inplace=True)]
        self.attr_layer = nn.Sequential(*attr_layers)

        self.attn_conv = nn.Sequential(
           DeformConv2d(in_channel + attr_channel, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.sa_layer =  nn.Sequential(
            SelfAttention(in_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mask_ = None

    def forward(self, x, attr_feature):
        attr_feature = attr_feature.unsqueeze(-1)
        attr_featureT = torch.transpose(attr_feature, 2, 3)
        attr_feature2d = torch.matmul(attr_feature, attr_featureT)
        attr = self.attr_layer(attr_feature2d)
        mask = self.attn_conv(torch.cat([x, attr], axis=1))  # [B, 1, H, W]
        x = x * mask
        x = self.sa_layer(x)
        return x

##PGmodel_module
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        eps = 1e-7
        mean = torch.mean(x**2, dim=1, keepdims=True)
        return x / (torch.sqrt(mean)+eps)
class Minibatch_std(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean = (x.var(dim=0) + 1e-8).sqrt().mean().view(1, 1, 1, 1)
        n, c, h, w = x.shape
        mean = torch.ones(n, 1, h, w, dtype=x.dtype, device=x.device)*mean
        return torch.cat((x, mean), dim=1)
class WeightScale(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, gain=2):
        scale = (gain/x.shape[1])**0.5
        return x * scale
class Conv2d(nn.Module):
    def __init__(self, inch, outch, kernel_size, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            WeightScale(),
            # nn.ZeroPad2d(padding),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(inch, outch, kernel_size, padding=0),
            PixelNorm(),
            )
        nn.init.kaiming_normal_(self.layers[2].weight)

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    Attn = Attention(64, 300, 1).to('cuda')
    from torchinfo import summary
    summary(Attn, input_size=[(64, 64, 64, 64), (64, 300, 128)])

