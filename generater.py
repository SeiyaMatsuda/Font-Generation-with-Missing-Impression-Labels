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
device = 'cuda'
import torch.nn as nn
class generated_image(nn.module):
    def __init__(self, G_model):
        self.model = G_model
    def
