import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(self, pretrained =True, num_classes=2, *args):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V2 if pretrained else None
        self.clf = resnet18(weights=weights)
        self.clf.fc = nn.Linear(512,num_classes)


    def forward(self, x, **kwargs):
        outputs = self.clf(x, **kwargs)
        return outputs

