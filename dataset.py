import os
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import Dataset

def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)

class CheXpertDataset(Dataset):
    def __init__(self, 
                 transform, 
                 data_dir='./data_frontal/',
                 num_samples=None,
                 seed=1
    ):
        self.data_dir = data_dir
        self.npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.num_samples = num_samples if num_samples is not None else len(self.npy_files)
        self.seed = seed
        
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

# Example usage
# Assuming you have torchvision transforms defined as needed
#transform = transforms.Compose([
#    transforms.ToTensor(),  # Adjust or add transforms as needed
#])

#dataset = CheXpertDataset(transform=transform, data_dir='./data_frontal/', num_samples=10000, seed=1)
#print(f"Loaded dataset with {len(dataset)} samples.")
