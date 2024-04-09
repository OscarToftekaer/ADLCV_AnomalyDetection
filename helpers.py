import numpy as np
import random
import torch
import matplotlib.pyplot as plt

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