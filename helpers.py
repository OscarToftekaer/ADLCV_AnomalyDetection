import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import torchvision
import os

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def im_normalize(im):
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn

def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))
    
def show(imgs, title=None, fig_titles=None, save_path=None): 

    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    fig, axs = plt.subplots(1, ncols=len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axs[i].imshow(img,cmap='gray')
        axs[i].axis('off')
        if fig_titles is not None:
            axs[i].set_title(fig_titles[i])

    if title is not None:
        plt.suptitle(title)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr,cmap='gray')
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

def prepare_dataloader(batch_size, img_size,data_dir,dataset_size, RGB = False):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from dataset import CheXpertDataset
    if RGB == False:
        transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,)),   # range [-1,1]
        transforms.Resize(size=(img_size,img_size))   #resizing to min dimensions
        ])
    else:
        transform = transforms.Compose([
            ToPILImageOrKeep(), 
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),                  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
            transforms.Resize(size=(img_size,img_size))   #resizing to min dimensions
        ])
    dataset = CheXpertDataset(transform, data_dir = data_dir, num_samples=dataset_size)
    # Split dataset into train and validation sets
    train_size = int(0.995 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def create_result_folders(experiment_name):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join('models', experiment_name), exist_ok=True)
    os.makedirs(os.path.join('results', experiment_name), exist_ok=True)
