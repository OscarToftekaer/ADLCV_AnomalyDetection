import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image

from helpers import prepare_dataloader
from Unet_model import UNet
from diffmodel import Diffusion



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 



diffusion = Diffusion(img_size=128, T=500, beta_start=1e-4, beta_end=0.02, device=device)



DATASET_SIZE = None
DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/diffusion_split'
batch_size = 1
img_size = 128

#load model
model = UNet(img_size=128, c_in=1, c_out=1, 
                time_dim=256, channels=32, device=device).to(device)
model.eval()
model.to(device)
model.load_state_dict(torch.load('models/ddpm/weights-30_v2.pt', map_location=device)) # load the given model



def fid_loop(batch_size, img_size, model, data_dir=DATA_DIR,dataset_size=DATASET_SIZE):
    t = torch.tensor(200).unsqueeze(0).to(device)
    _,val_loader = prepare_dataloader(batch_size, img_size,data_dir=DATA_DIR,dataset_size=DATASET_SIZE, RGB = True)
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    features_images = []
    features_samples = []

    for images in val_loader:
        with torch.no_grad():
            images = images.to(device)
            pred = inception(images.to(device))
            features_images.append(pred.detach().cpu().numpy())

            x_t, _ = diffusion.q_sample(images, t)   
            sampled_images = diffusion.ddim_sample_loop(model, x_t, 200, batch_size=images.shape[0])
            sampled_images = sampled_images.squeeze(0)
            pred = inception(sampled_images.to(device))
            features_samples.append(pred.detach().cpu().numpy())


    features_images = np.concatenate(features_images, axis=0)
    features_samples = np.concatenate(features_samples, axis=0)

    # Calculate mean and covariance
    mu1, sigma1 = features_images.mean(axis=0), np.cov(features_images, rowvar=False)
    mu2, sigma2 = features_samples.mean(axis=0), np.cov(features_samples, rowvar=False)

    # Calculate FID
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid


# Function to calculate Frechet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)




# L1 Loss between val images and samples
def l1_loss(images, samples):
    return torch.nn.functional.l1_loss(images, samples, reduction='mean')

# Mean Absolute Confidence Difference (MAD) between val images and samples
def MAD(images, samples):

    abs_diff = torch.abs(images - samples)
    MAD_value = torch.mean(abs_diff)

    return MAD_value.item()



def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,)),   # range [-1,1]
        transforms.Resize(size=(128,128))   #resizing to min dimensions
        ])
    # Open image
    img = Image.open(image_path)
    # Apply transformations
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img


# test of functions
images = load_image('results/ddpm/28_v2.jpg')
samples = load_image('results/ddpm/29_v2.jpg')
# print(l1_loss(images, samples))
# print(MAD(images, samples))


print(fid_loop(batch_size, img_size, model, data_dir=DATA_DIR,dataset_size=DATASET_SIZE))
