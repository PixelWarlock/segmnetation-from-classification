import torch
import torch.nn as nn

class FullModel(nn.Module):
    def __init__(self, 
                 segmentor:nn.Module,
                 classifier:nn.Module):
        super(FullModel, self).__init__()
        self.segmentor = segmentor
        self.classifier = classifier
    
    def forward(self, x):
        z = self.segmentor(x)
        k = torch.round(x) * z
        y = self.classifier(k)
        return y