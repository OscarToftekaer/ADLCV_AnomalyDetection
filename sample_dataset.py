import os
import numpy as np
import random
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
from helpers import save_images
from PIL import Image 
import torch

class samples_Dataset(Dataset):
    def __init__(self, 
                 transform, 
                 data_dir,
                 num_samples=None,
                 seed=1
    ):
        self.data_dir = data_dir
        self.images = []
        for folder in os.listdir(data_dir):
            file_path = os.path.join(data_dir, folder)
            for file in os.listdir(file_path):
                image_path = os.path.join(file_path, file)
                self.images = np.append(self.images,image_path)

        random.seed(seed)
        self.num_samples = num_samples if num_samples is not None else len(self.images)
        self.images = self.images[:self.num_samples]
    
        # Optionally, reduce dataset size
        if self.num_samples < len(self.images):
            self.images = random.sample(self.images, self.num_samples)
        
        print(f"Dataset size: {len(self.images)}")
        
        self.transform = transform
                
    # Return the number of images in the dataset
    def __len__(self):
        return self.num_samples
    
    # Get the image at a given index
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image  = Image.open(image_path)


        if self.transform:
            image = self.transform(image)
        else:
            # Ensure image is in the format CHW for PyTorch
            if image.ndim == 2:  # For grayscale images
                image = image[:, :, None]
            image = transforms.ToTensor()(image)
        grayscale_tensor = torch.mean(image, dim=0, keepdim=True)
        relative_path = os.path.relpath(image_path, self.data_dir)
        first_part_relative_path = relative_path.split('/')[0]  # Take only the first part

        assert first_part_relative_path in ['ill','healthy'], f"Unexpected class: {first_part_relative_path}"
        label = 0 if first_part_relative_path == 'healthy' else 1

        return grayscale_tensor, label

# Example usage
if __name__ == "__main__":
    import tqdm
    import matplotlib.pyplot as plt
    from clf_dataset import CLF_Dataset
    import torch
    DATASET_SIZE = None
    #DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/classification_split'
    DATA_DIR = '/work3/s194632/samples_ddim'
    IMG_SIZE = 128
    BATCH_SIZE = 1
    LR = 1e-3
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=10),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE))
    ])
    dataset = samples_Dataset(transform=transform, data_dir=DATA_DIR)
    #dataset = CLF_Dataset(transform=transform, data_dir=DATA_DIR)

    for i in range(1):
        idx = i # Index of the instance you want to get the npy_path for
        image,label = dataset[idx]

        print(image.shape)
        print(label)
        save_images(image,'./results/clf/samples_'+str(i))