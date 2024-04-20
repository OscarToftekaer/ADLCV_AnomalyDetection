import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super(ResNet18, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.clf = resnet18(weights=weights)
        
        # Modify the first convolutional layer to accept one channel
        original_first_layer = self.clf.conv1
        self.clf.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # If pretrained, initialize the modified first layer by averaging the original weights
        if pretrained:
            with torch.no_grad():
                # Average across the input channel dimension and assign to the new first layer
                self.clf.conv1.weight.data = original_first_layer.weight.data.mean(dim=1, keepdim=True)
        
        # Replace the fully connected layer to match the number of classes
        self.clf.fc = nn.Linear(self.clf.fc.in_features, num_classes)

    def forward(self, x, **kwargs):
        outputs = self.clf(x, **kwargs)
        return outputs

