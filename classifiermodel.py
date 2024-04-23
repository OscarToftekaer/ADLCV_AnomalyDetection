import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
# TO DO: change so it doesn't average over first weights but dublicates the channels in stead.  

class ResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(ResNet18, self).__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.clf = resnet18(weights=weights)
        
        # Modify the first convolutional layer to accept one channel
        self.clf.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.clf.fc = nn.Linear(self.clf.fc.in_features, num_classes)

    def forward(self, x, **kwargs):
        outputs = self.clf(x, **kwargs)
        return outputs

