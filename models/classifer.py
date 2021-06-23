import torch
import torch.nn as nn
import torch.nn.functional as F
import word2vec
class classifier(nn.Module):
    def __init__(self, num_class, img_size, num_impression_word):
        super(classifier,self).__init__()
        self.num_dimension = num_class
        self.img_size = img_size
        self.num_impression_word = num_impression_word
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_class, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2,  inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, self.num_impression_word),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1,14*14*512)
        x = self.fc1(x)
        x = self.fc2(x)

        return x