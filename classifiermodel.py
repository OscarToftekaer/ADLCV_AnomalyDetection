import torch
import torch.nn as nn
from transformers import ResNet18Config, ResNet18Model

# Define a class for the ResNet18 model with fine-tuning
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # Load the pre-trained ResNet18 model
        config = ResNet18Config.from_pretrained("resnet18")
        self.resnet = ResNet18Model(config)
        # Replace the classification head for our custom task
        self.resnet.fc = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, **kwargs):
        outputs = self.resnet(input_ids, **kwargs)
        return outputs

# Initialize the model
num_classes = 10  # Number of classes in your custom task
model = ResNet18(num_classes)

# Fine-tune all layers
for param in model.parameters():
    param.requires_grad = True
