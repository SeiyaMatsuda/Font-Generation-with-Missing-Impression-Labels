import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from sklearn.decomposition import PCA
import torch.nn.utils as utils
from math import sqrt
import random
# from .condbatchnorm import CondBatchNorm2d
import torch.nn.functional as F
# from .self_attention import Self_Attn
from torch.nn.parameter import Parameter
from models.Deepsets import DeepSets
def weights_init(m):
    if isinstance(m, CustomConv2d):
        if m.conv.weight is not None:
            if m.residual_init:
                init.xavier_uniform_(m.conv.weight.data, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(m.conv.weight.data)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias.data, 0.0)
    if isinstance(m, CustomLinear):
        if m.linear.weight is not None:
            init.xavier_uniform_(m.linear.weight.data)
        if m.linear.bias is not None:
            init.constant_(m.linear.bias.data, 0.0)
    if isinstance(m, CustomEmbedding):
        if m.embed.weight is not None:
            init.xavier_uniform_(m.embed.weight.data)


def global_pooling(input, pooling='mean'):
    if pooling == 'mean':
        return input.mean(3).mean(2)
    elif pooling == 'sum':
        return input.sum_weight(3).sum_weight(2)
    else:
        raise NotImplementedError()


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


class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, spectral_norm=False):
        super(CustomEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        if spectral_norm:
            self.embed = utils.spectral_norm(self.embed)

    def forward(self, input):
        return self.embed(input)


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


class MeanPoolConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 spectral_norm=False,
                 residual_init=True):
        super(MeanPoolConv, self).__init__()
        self.conv = CustomConv2d(in_channels,
                                 out_channels,
                                 kernel_size,
                                 bias=bias,
                                 spectral_norm=spectral_norm,
                                 residual_init=residual_init)

    def forward(self, input):
        output = input
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_square = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, in_height, in_width, in_depth) = output.size()
        out_depth = int(in_depth / self.block_size_square)
        out_width = int(in_width * self.block_size)
        out_height = int(in_height * self.block_size)
        output = output.contiguous().view(batch_size, in_height, in_width,
                                          self.block_size_square, out_depth)
        output_list = output.split(self.block_size, 3)
        output_list = [
            output_element.contiguous().view(batch_size, in_height, out_width,
                                             out_depth)
            for output_element in output_list
        ]
        output = torch.stack(output_list, 0).transpose(0, 1).permute(
            0, 2, 1, 3, 4).contiguous().view(batch_size, out_height, out_width,
                                             out_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 spectral_norm=False,
                 residual_init=True):
        super(UpSampleConv, self).__init__()
        self.conv = CustomConv2d(in_channels,
                                 out_channels,
                                 kernel_size,
                                 bias=bias,
                                 spectral_norm=spectral_norm,
                                 residual_init=residual_init)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
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


class OptimizedResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 spectral_norm=False):
        super(OptimizedResidualBlock, self).__init__()
        self.conv1 = CustomConv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  spectral_norm=spectral_norm)
        self.conv2 = ConvMeanPool(out_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  spectral_norm=spectral_norm)
        self.conv_shortcut = MeanPoolConv(in_channels,
                                          out_channels,
                                          kernel_size=1,
                                          spectral_norm=spectral_norm,
                                          residual_init=False)
        self.relu2 = nn.ReLU()

    def forward(self, input):
        shortcut = self.conv_shortcut(input)

        output = input
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output


class CondResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_classes,
                 resample=None,
                 spectral_norm=False):
        super(CondResidualBlock, self).__init__()
        if in_channels != out_channels or resample is not None:
            self.learnable_shortcut = True
        else:
            self.learnable_shortcut = False

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'up':
            self.norm1 = CondBatchNorm2d(in_channels, num_classes)
            self.norm2 = CondBatchNorm2d(out_channels, num_classes)
            self.conv_shortcut = UpSampleConv(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              spectral_norm=spectral_norm,
                                              residual_init=False)
            self.conv1 = UpSampleConv(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
            self.conv2 = CustomConv2d(out_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      spectral_norm=spectral_norm)
        else:
            raise NotImplementedError()

    def forward(self, input, label=None, class_weight=None):
        if label is None and class_weight is None:
            raise ValueError('either label or class_weight must not be None')

        if self.learnable_shortcut:
            shortcut = self.conv_shortcut(input)
        else:
            shortcut = input

        output = input
        output = self.norm1(output, label=label, class_weight=class_weight)
        output = self.relu1(output)
        output = self.conv1(output)
        output = self.norm2(output, label=label, class_weight=class_weight)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output

class ImpEmbedding(nn.Module):
    def __init__(self, weight, sum_weight = True, deepsets = False,  num_dimension = 300, residual_num = 0, required_grad = False):
        super(ImpEmbedding, self).__init__()
        self.weight = weight
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
            labels = labels.view(labels.size(0), labels.size(1), 1)
            attr = torch.mul(self.embed.weight.data, labels)
            # attr = list(map(lambda x:x[x.sum(2)!=0], attr.split(1)))
            # attr = torch.cat(list(map(lambda x:x[torch.tensor(random.choices(range(len(x)), k=64))], attr)), dim=0)
            # attr = attr.view(-1, 64, 300)
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
class Down(nn.Module):
    def __init__(self, in_channel, out_channel, normalize=True, attention=False,
                 lrelu = False, dropout=0.0, bias=False, kernel_size=4, stride=2, padding=1):
        super(Down, self).__init__()
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias)]
        if attention:
            layers.append(Self_Attn(out_channel))
        if normalize:
            layers.append(nn.BatchNorm2d(out_channel))
        if lrelu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
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

class EqualizedLR_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = np.sqrt(2 / (in_ch * kernel_size[0] * kernel_size[1]))

        self.weight = Parameter(torch.Tensor(out_ch, in_ch, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_ch))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class Pixel_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        b = a / torch.sqrt(torch.sum(a ** 2, dim=1, keepdim=True) + 10e-8)
        return b
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
        eps = 1e-7
        std = torch.std(x + eps, dim=0, keepdim=True)
        mean = torch.mean(std, dim=(1,2,3), keepdim=True)
        n,c,h,w = x.shape
        mean = torch.ones(n,1,h,w, dtype=x.dtype, device=x.device)*mean
        return torch.cat((x,mean), dim=1)
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

