from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetS11(nn.Module):
    def __init__(self):
        super(ResnetS11, self).__init__()
        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(dropout_rate)
        )
        # layer1
        self.layer1Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(dropout_rate)
        )
        self.layer1resnetBlock1 = self.resnetBlock(128, 128)
        # layer2
        self.layer2Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(dropout_rate)
        )
        # layer3
        self.layer3Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.Dropout(dropout_rate)
        )
        self.layer3resnetBlock1 = self.resnetBlock(512, 512)
        # ending layer or layer-4
        self.maxpool = nn.MaxPool2d(4, 4)

        self.fc_layer = nn.Linear(512,10)

    def resnetBlock(self, in_channels, out_channels):
        l = []
        l.append(nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=False))
        l.append(nn.BatchNorm2d(out_channels))
        l.append(nn.ReLU())
        l.append(nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=False))
        l.append(nn.BatchNorm2d(out_channels))
        l.append(nn.ReLU())
        return nn.Sequential(*l)

    def forward(self, x):
        # prepLayer
        x = self.prepLayer(x)
        # Layer1
        x = self.layer1Conv1(x)
        r1 = self.layer1resnetBlock1(x)
        x = torch.add(x, r1)
        # layer2
        x = self.layer2Conv1(x)
        # layer3
        x = self.layer3Conv1(x)
        r2 = self.layer3resnetBlock1(x)
        x = torch.add(x, r2)
        # layer4 or ending layer
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)