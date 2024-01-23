import torch.nn as nn

class FullModel(nn.Module):
    def __init__(self, 
                 segmentor:nn.Module,
                 classifier:nn.Module):
        super(FullModel, self).__init__()
        self.segmentor = segmentor
        self.classifier = classifier
    
    def forward(self, x):
        x = self.segmentor(x)
        x = self.classifier(x)
        return x