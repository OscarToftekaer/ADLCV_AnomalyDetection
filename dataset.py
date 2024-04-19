import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
import blobfile as bf

def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)

class CheXpertDataset(Dataset):
    def __init__(self, 
                 transform, 
                 data_dir='/dtu/datasets1/ashery-chexpert/data/diffusion_split',
                 num_samples=None,
                 seed=1
    ):
        # self.data_dir = data_dir
        # self.npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        # self.num_samples = num_samples if num_samples is not None else len(self.npy_files)
        # self.seed = seed

        self.data_dir = data_dir
        self.npy_files = []
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            for study_folder in os.listdir(folder_path):
                study_folder_path = os.path.join(folder_path, study_folder)
                for file in os.listdir(study_folder_path):
                        file_path = os.path.join(study_folder_path, file)
                        for file in os.listdir(file_path):
                            npy_path = os.path.join(file_path, file)
                            self.npy_files = np.append(self.npy_files,npy_path)


        random.seed(seed)
        random.shuffle(self.npy_files)
        self.num_samples = num_samples if num_samples is not None else len(self.npy_files)
        self.npy_files = self.npy_files[:self.num_samples]
    

        # Optionally, reduce dataset size
        if self.num_samples < len(self.npy_files):
            set_seed(seed=self.seed)
            self.npy_files = random.sample(self.npy_files, self.num_samples)
        
        print(f"Dataset size: {len(self.npy_files)}")
        
        self.transform = transform
                
    # Return the number of images in the dataset
    def __len__(self):
        return self.num_samples
    
    # Get the image at a given index
    def __getitem__(self, idx):
        npy_path = self.npy_files[idx]
        image = np.load(npy_path)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Ensure image is in the format CHW for PyTorch
            if image.ndim == 2:  # For grayscale images
                image = image[:, :, None]
            image = transforms.ToTensor()(image)
        
        return image

