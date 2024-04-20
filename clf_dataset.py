import os
import numpy as np
import random
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
from helpers import save_images

class CLF_Dataset(Dataset):
    def __init__(self, 
                 transform, 
                 data_dir,
                 num_samples=None,
                 seed=1
    ):

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

                # Extract relative path
        relative_path = os.path.relpath(npy_path, self.data_dir)
        first_part_relative_path = relative_path.split('/')[0]  # Take only the first part
        
        # Assign label based on the first part of the relative path
        assert first_part_relative_path in ['cla_NPMhealthy', 'cla_PMill'], f"Unexpected class: {first_part_relative_path}"
        label = 0 if first_part_relative_path == 'cla_NPMhealthy' else 1
        
        return image, label

# Example usage
if __name__ == "__main__":
    import tqdm
    import matplotlib.pyplot as plt
    DATASET_SIZE = None
    DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/classification_split'
    IMG_SIZE = 256
    BATCH_SIZE = 1
    LR = 1e-3
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE))
    ])
    dataset = CLF_Dataset(transform=transform, data_dir=DATA_DIR)


    for i in range(20):
        idx = i # Index of the instance you want to get the npy_path for
        image, label = dataset[idx]
        print(label)
        save_images(image,'./results/clf/test_'+str(i))