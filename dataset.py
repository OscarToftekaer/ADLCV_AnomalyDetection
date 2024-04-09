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
# transform = transforms.Compose([
#    transforms.ToTensor(),  # Adjust or add transforms as needed
# ])

#dataset = CheXpertDataset(transform=transform, data_dir='./data_frontal/', num_samples=10000, seed=1)
#print(f"Loaded dataset with {len(dataset)} samples.")

if __name__ == "__main__":

    from diffmodel import Diffusion
    from classifiermodel import UNet
    from helpers import im_normalize, tens2image, show
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 2929 

    os.makedirs('assets/', exist_ok=True)
    
    # dataset and dataloaders
    transform = transforms.Compose([
            transforms.ToTensor(),   
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),           
            transforms.Resize((256, 256))         # resize to 16x16
            ])

    diffusion = Diffusion(device=device)
    transform = transforms.Compose([
    transforms.ToTensor(),  # Adjust or add transforms as needed
    ])

    batch_size = 1

    dataset = CheXpertDataset(transform=transform)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    batch_images  = next(iter(trainloader))
        
    for i in range(len(batch_images)):
        image = batch_images[i]  # Assuming batch size is 1
        plt.imshow(np.transpose(image, (1, 2, 0)),cmap='gray')
        plt.axis("off")
        plt.show()
    

    #timesteps for forward
    t = torch.Tensor([0, 50, 100, 150, 200, 300, 499]).long().to(device)
    fig_titles = [f'Step {ti.item()}' for ti in t]
    x0 = image.unsqueeze(0).to(device) 
    xt, noise = diffusion.q_sample(x0, t)
    #####################################################

    noised_images = np.stack([im_normalize(tens2image(xt[idx].cpu())) for idx in range(t.shape[0])], axis=0)
    show(noised_images, title='Forward process', fig_titles=fig_titles, save_path='assets/forward.png')

    model = UNet(device=device)
    model.eval()