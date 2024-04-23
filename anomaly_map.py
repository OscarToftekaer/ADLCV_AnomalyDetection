import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import wandb
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


from Unet_model import UNet
from diffmodel import Diffusion




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#load model
model = UNet(img_size=128, c_in=1, c_out=1, 
                time_dim=256, channels=32, device=device).to(device)
model.eval()
model.to(device)
model.load_state_dict(torch.load('models/ddpm/weights-40_v2.pt', map_location=device)) # load the given model

diffusion = Diffusion(img_size=128, T=500, beta_start=1e-4, beta_end=0.02, device=device)


def make_anomaly_map(path_to_img, model):

    images = np.load(path_to_img)

    resize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),                 
            transforms.Resize(size=(128,128))   
            ])

    images = resize_transform(images).to(device)

    t_step = 150
    t = torch.tensor(t_step).unsqueeze(0).to(device)
    x_t, _ = diffusion.q_sample(images, t)   
    sampled_images = diffusion.ddim_sample_loop(model, x_t, t_step, batch_size=images.shape[0])
    sampled_images = sampled_images.squeeze(0)
    images = images.squeeze(0)
    diff_images = (images - sampled_images).cpu().numpy()
    thresh = torch.tensor(threshold_otsu(diff_images))
    diff_images = torch.where(torch.tensor(diff_images) > thresh, 1, 0)  #this is our predicted binary segmentation

    diff_images = diff_images.permute(1, 2, 0)




    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    # Original image
    axes[0].imshow(images.cpu().numpy().squeeze(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Noised image
    axes[1].imshow(x_t.cpu().numpy().squeeze(), cmap='gray')
    axes[1].set_title('Noised Image')
    axes[1].axis('off')

    # Sampled image
    axes[2].imshow(sampled_images.cpu().numpy().squeeze(), cmap='gray')
    axes[2].set_title('Sampled Image')
    axes[2].axis('off')

    # Anomaly map
    axes[3].imshow(diff_images.cpu().numpy().squeeze(), cmap='copper')
    axes[3].set_title('Anomaly Map (Pixel-wise Difference)')
    axes[3].axis('off')


    plt.tight_layout()
    plt.savefig('test_map_40.jpg')

    return


path_to_img = '/dtu/datasets1/ashery-chexpert/data/inference_split/inf_PMill/patient00236/study2/view1_frontal.npy'
make_anomaly_map(path_to_img, model)
