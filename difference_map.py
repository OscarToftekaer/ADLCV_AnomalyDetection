import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import wandb
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from helpers import create_result_folders,prepare_dataloader,save_images
import os


from Unet_model import UNet
from diffmodel import Diffusion


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#load model
model = UNet(img_size=128, c_in=1, c_out=1, 
                time_dim=256, channels=32, device=device).to(device)
model.eval()
model.to(device)
model.load_state_dict(torch.load('models/ddpm/weights-50_v2.pt', map_location=device)) # load the given model

diffusion = Diffusion(img_size=128, T=500, beta_start=1e-4, beta_end=0.02, device=device)


def make_anomaly_map(image_paths, model):
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 6 * len(image_paths)))

    for idx, path_to_img in enumerate(image_paths):
        images = np.load(path_to_img)

        resize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),                 
            transforms.Resize(size=(128, 128))   
        ])
        resize_transform_sample = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        images = resize_transform(images).to(device)

        t_step = 300
        t = torch.tensor(t_step).unsqueeze(0).to(device)
        x_t, _ = diffusion.q_sample(images, t)   
        sampled_images =  diffusion.ddim_sample_loop(model, x_t, t_step, batch_size=images.shape[0])
        sampled_images = sampled_images.cpu().numpy().squeeze()
        images = images.squeeze()
        sampled_images = resize_transform_sample(sampled_images).to(device)

        difference = abs((images - sampled_images).cpu().numpy())

        otsu_mask = difference
        difference = np.transpose(difference, (1, 2, 0)) 
        thresh = torch.tensor(threshold_otsu(otsu_mask))
        print(thresh)
        otsu_mask = torch.where(torch.tensor(otsu_mask) > 0.05, 1, 0)  #this is our predicted binary segmentation

        otsu_mask = otsu_mask.permute(1, 2, 0)

        # Original image
        img_original = axes[idx, 0].imshow(images.cpu().numpy().squeeze(), cmap='gray')
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')
        # fig.colorbar(img_original, ax=axes[idx, 0])

        # Sampled image
        img_sampled = axes[idx, 1].imshow(sampled_images.cpu().numpy().squeeze(), cmap='gray')
        axes[idx, 1].set_title('Sampled Image')
        axes[idx, 1].axis('off')
        # fig.colorbar(img_sampled, ax=axes[idx, 1])

        # Difference map
        img_difference = axes[idx, 2].imshow(difference.squeeze(), cmap='Blues')
        axes[idx, 2].set_title('Difference Map')
        axes[idx, 2].axis('off')
        # fig.colorbar(img_difference, ax=axes[idx, 2])

        # Otsu threshold
        img_otsu = axes[idx, 3].imshow(otsu_mask.cpu().numpy().squeeze(), cmap='copper')
        axes[idx, 3].set_title('Otsu threshold')
        axes[idx, 3].axis('off')
        # fig.colorbar(img_otsu, ax=axes[idx, 3])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # Adjust the vertical spacing between rows
    plt.savefig('difference_maps_t=300.jpg')


image_paths = ['/dtu/datasets1/ashery-chexpert/data/inference_split/inf_PMill/patient00236/study2/view1_frontal.npy',
               '/dtu/datasets1/ashery-chexpert/data/inference_split/inf_PMill/patient05029/study2/view1_frontal.npy',
               '/dtu/datasets1/ashery-chexpert/data/inference_split/inf_PMill/patient11421/study1/view1_frontal.npy',
               '/dtu/datasets1/ashery-chexpert/data/inference_split/inf_PMill/patient15474/study1/view1_frontal.npy']
make_anomaly_map(image_paths, model)
