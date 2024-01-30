import torch
import torch.nn as nn
""""""
class Classifier(nn.Module):
    def __init__(self, num_classes, in_channels:int=1):
        super(Classifier, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.activation = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1)

        self.linear = nn.Linear(in_features=392,out_features=num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.activation(x)
        return x


def get_efficientnet_b0(num_classes):
    backbone = torch.hub.load(
            'rwightman/gen-efficientnet-pytorch',
            'efficientnet_b1',
            pretrained=True,
        )
    backbone = nn.Sequential(*list(backbone.as_sequential())[:-1]) 
    model = nn.Sequential(*[
        backbone,
        nn.Linear(in_features=1280, out_features=num_classes, bias=True),
        nn.Softmax(dim=1)
    ])
    return model