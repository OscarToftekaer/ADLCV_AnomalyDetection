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

from diffmodel import Diffusion
from Unet_model import UNet

from torchvision.datasets import MNIST


DATASET_SIZE = None
DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/diffusion_split'
with_logging = False

img_size = 64



def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

def prepare_dataloader(batch_size, img_size):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from dataset import CheXpertDataset
    transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,)),   # range [-1,1]
    transforms.Resize(size=(img_size,img_size))   #resizing to min dimensions
    ])
    dataset = CheXpertDataset(transform, data_dir = DATA_DIR, num_samples=DATASET_SIZE)
    # dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def create_result_folders(experiment_name):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join('models', experiment_name), exist_ok=True)
    os.makedirs(os.path.join('results', experiment_name), exist_ok=True)

def train(device='cuda', T=500, img_size=img_size, input_channels=1, channels=32, time_dim=256,
          batch_size=1, lr=1e-3, num_epochs=10, experiment_name='ddpm', show=False):
    '''Implements algrorithm 1 (Training) from the ddpm paper at page 4'''
    print('entering traning')
    create_result_folders(experiment_name)
    dataloader = prepare_dataloader(batch_size, img_size)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 time_dim=time_dim,channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss() # use MSE loss 
    
    l = len(dataloader)

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(dataloader)

        for i, images in enumerate(pbar):
            images = images.to(device)

            # test of q sample images
            # tt = torch.randint(low=100, high=101, size=(images.shape[0],), device=device)
            # x_t, _ = diffusion.q_sample(images, tt)    
            # save_images(images=x_t, path=os.path.join('results', experiment_name, f'{epoch}_qsample_100.jpg'),
            # show=show, title=f'Epoch {epoch}')
        

            t = diffusion.sample_timesteps(images.shape[0]).to(device) # line 3 from the Training algorithm
            x_t, noise = diffusion.q_sample(images, t) # inject noise to the images (forward process), HINT: use q_sample
            predicted_noise = model(x_t, t) # predict noise of x_t using the UNet
            loss = mse(noise, predicted_noise) # loss between noise and predicted noise

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pbar.set_postfix(MSE=loss.item())

        if with_logging:
            wandb.log({"loss": loss
                    })
        
        # add noise to images again
        tt = torch.randint(low=50, high=51, size=(images.shape[0],), device=device)
        x_t, _ = diffusion.q_sample(images, tt)    
        sampled_images = diffusion.ddim_sample_loop(model, x_t, batch_size=images.shape[0])
        save_images(images=sampled_images, path=os.path.join('results', experiment_name, f'{epoch}.jpg'),
                   show=show, title=f'Epoch {epoch}')
        torch.save(model.state_dict(), os.path.join('models', experiment_name, f'weights-{epoch}.pt'))
        print('end')




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f'Model will run on {device}')
        # Initialize logging
    if with_logging:
        print("with logging")
            
        wandb.init(
        project="ADLCV_AnomalyDetection", entity="ADLCV_exam_project",
        config={
        "device": device,
        "architecture": "UNet"
        }
    )
    train(device=device)
    print('finito')

    

        