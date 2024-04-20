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
from helpers import create_result_folders,prepare_dataloader,save_images

with_logging = False

DATASET_SIZE = None
DATA_DIR = '/dtu/datasets1/ashery-chexpert/data/classification_split'
IMG_SIZE = 64
BATCH_SIZE = 1
LR = 1e-3
NUM_EPOCHS = 10

def clf_train(device = 'cuda', img_size = IMG_SIZE, batch_size = BATCH_SIZE):
