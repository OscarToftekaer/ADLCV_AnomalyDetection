import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
import wandb

import torchvision
from tqdm import tqdm
from torch import optim

from classifiermodel import ResNet18
from helpers import create_result_folders,prepare_dataloader,save_images
import torchvision.transforms as transforms
from clf_dataset import CLF_Dataset
from torch.utils.data import DataLoader

with_logging = False

DATASET_SIZE = None
DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/diffusion_split'
IMG_SIZE = 128
BATCH_SIZE = 2
LR = 1e-4
NUM_EPOCHS = 20

def clf_train(device = 'cuda', img_size = IMG_SIZE, batch_size = BATCH_SIZE, lr = LR, num_epochs = NUM_EPOCHS, experiment_name = 'clf', show = False):
    print('Classifier training starts')
    create_result_folders(experiment_name)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(13, 9), sigma=(0.25, 5.)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE))
    ])

    # Initialize dataset
    dataset = CLF_Dataset(transform=transform, data_dir=DATA_DIR)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model
    model = ResNet18(pretrained=True, num_classes=2)  # Assuming 2 classes (healthy, ill)

    for param in model.parameters():
        param.requires_grad = True
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01,momentum = 0.9, weight_decay=1e-2)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,step_size_up=500, base_lr=LR, max_lr=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)  # Assuming the output of the model is logits
            
            loss.backward()
            optimizer.step()
            #scheduler.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)  # Assuming the output of the model is logits
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        

        # Check if this is the best epoch so far
        if val_loss < best_val_loss :#and val_loss < train_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save the model checkpoint
            torch.save(model.state_dict(), 'models/clf/small_batch_diagnostic_best_model_checkpoint.pth')
            print("Best model checkpoint saved!")

    print(f"Training finished. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}.")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f'Model will run on {device}')
        # Initialize logging
    if with_logging:
        print("with logging")
            

    clf_train(device=device)
    print('finito')





