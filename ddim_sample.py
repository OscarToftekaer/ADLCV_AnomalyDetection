import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from helpers import create_result_folders, prepare_dataloader, save_images
import os
from Unet_model import UNet
from diffmodel import Diffusion
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = UNet(img_size=128, c_in=1, c_out=1,
             time_dim=256, channels=32, device=device).to(device)
model.eval()
model.load_state_dict(torch.load('models/ddpm/weights-105.pt', map_location=device))  # Load the given model

diffusion = Diffusion(img_size=128, T=500, beta_start=1e-4, beta_end=0.02, device=device)
DATASET_SIZE = None
DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/inference_split'

img_size = 128
batch_size = 1

train_loader, _ = prepare_dataloader(batch_size, img_size, data_dir=DATA_DIR, dataset_size=DATASET_SIZE)

# Create folder to save sampled images
result_folder = "samples_ddim"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

resize_transform_sample = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

pbar = tqdm(train_loader)

for i, images in enumerate(pbar):
    images = images.to(device)

    t_step = 300
    t = torch.tensor(t_step).unsqueeze(0).to(device)
    x_t, _ = diffusion.q_sample(images, t)
    sampled_images = diffusion.ddim_sample_loop(model, x_t, t_step, batch_size=images.shape[0])
    sampled_images = sampled_images.cpu().numpy().squeeze()
    sampled_images = resize_transform_sample(sampled_images).to(device)


    sampled_images = np.transpose(sampled_images.to('cpu').numpy(), (1, 2, 0)) 
    sample_img = np.squeeze(sampled_images)

    plt.imsave(os.path.join(result_folder, f"sample_{i}_.png"), sample_img, cmap='gray')



