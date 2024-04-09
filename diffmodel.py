import torch
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
import math
import numpy as np
import torch.nn.functional as F

class Diffusion:
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, img_size=256,device="cuda"):
        """
        T : total diffusion steps (X_T is pure noise N(0,1))
        beta_start: value of beta for t=0
        b_end: value of beta for t=T

        """

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.betas = self.get_betas().to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0) # cumulative products of alpha
        self.alphas_bar_prev = torch.cat((torch.ones(1), self.alphas_bar[:-1]),0)
        self.alphas_bar_next = torch.cat((self.alphas_bar[1:], torch.zeros(1)),0)


    def get_betas(self, schedule='linear'):
        if schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.T)
        else :
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5,self.T) **2
    
    def q_sample(self, x, t):
        """
        x: input image (x0)
        t: timestep: should be torch.tensor

        Forward diffusion process
        q(x_t | x_0) = sqrt(alpha_hat_t) * x0 + sqrt(1-alpha_hat_t) * N(0,1)

        Returns q(x_t | x_0), noise
        """
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t])
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None] # match image dimensions
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_bar[t])
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None, None]# match image dimensions
        
        noise = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise
    

    def x0_pred(self,model,x_t, t):
        """
        Predict x0 from x_t from epsilon as 

        x0 = sqrt(1/alpha_t) * x_t - sqrt(1/alpha_t-1) * epsilon
        """
        # TODO: Check the dimensions of the tensors of x_t 
        eps = model(x_t, t)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_bar[t])[:, None, None, None]
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_bar[t] - 1)[:, None, None, None]
        x0 = sqrt_recip_alphas_cumprod * x_t + sqrt_recipm1_alphas_cumprod * eps

        return x0

    def _predict_eps_from_xstart(self,model, x_t, t):

        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_bar[t])[:, None, None, None]
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_bar[t] - 1)[:, None, None, None]
        x0 = self.x0_pred(model, x_t, t)

        return (sqrt_recip_alphas_cumprod * x_t - x0) / sqrt_recipm1_alphas_cumprod

      
    def ddim_sample(self, model, x_t, t):
        """
        Sample from p(x{t-1} | x_t) using the reverse process and model
        """
        eps = self._predict_eps_from_xstart(model, x_t, t)
        x0 = self.x0_pred(model, x_t, t)
                
                # Equation 12.
        return x0 * torch.sqrt(self.alphas_bar_prev) + torch.sqrt(1-self.alphas_bar_prev) * eps
    
    def ddim_reverse_sample(self, model, x_t,t):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        eps = self._predict_eps_from_xstart(model, x_t, t)
        x0 = self.x0_pred(model, x_t, t)

                # Equation 12. reversed
        return x0 * torch.sqrt(self.alphas_bar_next) + torch.sqrt(1-self.alphas_bar_next) * eps


    def ddim_sample_loop(self, model, batch_size, timesteps_to_save=None, verbose=True):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """ 
        """
        y is class label
        """
        if verbose:
            logging.info(f"Sampling {batch_size} new images....")
            pbar = tqdm(reversed(range(1, self.T)), position=0, total=self.T-1)
        else :
            pbar = reversed(range(1, self.T))
            
        model.eval()
        if timesteps_to_save is not None:
            intermediates = []
        with torch.no_grad():
            x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)
            for i in pbar:
                t = (torch.ones(batch_size) * i).long().to(self.device)
                # T-1, T-2, .... 0
                x = self.ddim_sample(model, x, t)
                if timesteps_to_save is not None and i in timesteps_to_save:
                    x_itermediate = (x.clamp(-1, 1) + 1) / 2
                    x_itermediate = (x_itermediate * 255).type(torch.uint8)
                    intermediates.append(x_itermediate)

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        if timesteps_to_save is not None:
            intermediates.append(x)
            return x, intermediates
        else :
            return x
    

    def sample_timesteps(self, batch_size, upper_limit=None):
        """
        Sample timesteps uniformly for training
        """
        if upper_limit is None:
            return torch.randint(low=1, high=self.T, size=(batch_size,), device=self.device)
        else :
            return torch.randint(low=1, high=upper_limit, size=(batch_size,), device=self.device)

