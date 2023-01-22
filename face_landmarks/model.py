import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# input size = 48 x 48
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 136)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

# input size = 224 x 224
class ResNet18(nn.Module):

    """
        Parameters
        ----------
        pretrained_weights : bool, default = "True"
            Ways of weights initialization. 
            If "False", it means random initialization and no pretrained weights,
            If "True" it means resnet34 pretrained weights are used.

        fine_tune: bool, default = "False"
            Allows to choose between two types of transfer learning: fine tuning and feature extraction.
            For more details of the description of each mode, 
            read https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

        embedding_size: int, default = 128
            Size of the embedding of the last layer
            
    """

    def __init__(self, pretrained_weights=True, fine_tune=False, embedding_size=136):
        super(ResNet18, self).__init__()
        self.pretrained_weights = pretrained_weights
        self.fine_tune = fine_tune
        
        if self.pretrained_weights:
            pretrained_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        else:
            pretrained_model = torchvision.models.resnet18(weights=None)
        
        if not self.fine_tune:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        pretrained_model.fc = torch.nn.Linear(pretrained_model.fc.in_features, embedding_size)
        pretrained_model = pretrained_model.type(torch.FloatTensor)

        self.model = pretrained_model

    def forward(self, x):
        x = self.model(x)
        return x

# input size = 128 x 128
class YinNet(nn.Module):
    def __init__(self):
        '''
        Взято у: https://github.com/yinguobing/cnn-facial-landmark/blob/master/landmark.py
        '''
        super(YinNet, self).__init__()
 
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), #conv1
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,stride=1), #conv2
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3,stride=1), #conv3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,stride=1), #conv4
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3,stride=1), #conv5
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1), #conv6
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3,stride=1), #conv7
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride=1), #conv8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.fc1 = nn.Linear(in_features=6400, out_features=1024)
        self.bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 136)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.fc2(x)
        
        return x