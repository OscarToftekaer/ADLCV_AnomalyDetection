import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image

from helpers import prepare_dataloader

DATASET_SIZE = None
DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/diffusion_split'
batch_size = 1
img_size = 128

_,val_loader = prepare_dataloader(batch_size, img_size,data_dir=DATA_DIR,dataset_size=DATASET_SIZE)



# L1 Loss between val images and samples
def l1_loss(images, samples):
    return torch.nn.functional.l1_loss(images, samples, reduction='mean')

# Mean Absolute Confidence Difference (MAD) between val images and samples
def MAD(images, samples):

    abs_diff = torch.abs(images - samples)
    MAD_value = torch.mean(abs_diff)

    return MAD_value.item()


# FID score
def calculate_fid(real_images, fake_images, device):
    # Load pretrained Inception model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    # Calculate features
    real_features = get_features(real_images, inception, device)
    fake_features = get_features(fake_images, inception, device)

    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

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

# Function to get features from inception
def get_features(images, model, device, batch_size=32):
    features = []
    for batch in val_loader:
        with torch.no_grad():
            pred = model(batch.to(device))
            features.append(pred.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


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
print(l1_loss(images, samples))
print(MAD(images, samples))
