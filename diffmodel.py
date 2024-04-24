import torch
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
import math
import numpy as np
import torch.nn.functional as F

class Diffusion:
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        """
        T : total diffusion steps (X_T is pure noise N(0,1))
        beta_start: value of beta for t=0
        b_end: value of beta for t=T

        """

        self.T = 500
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.betas = self.get_betas().to(device)
        self.alphas = (1. - self.betas).to(self.device)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(device) # cumulative products of alpha
        # self.alphas_bar_prev = torch.cat((torch.ones(1), self.alphas_bar[:-1]),0).to(device)
        self.alphas_bar_prev = torch.cat((torch.ones(1, device=device), self.alphas_bar[:-1]), 0)

        # self.alphas_bar_next = torch.cat((self.alphas_bar[1:], torch.zeros(1)),0).to(device)
        self.alphas_bar_next = torch.cat((self.alphas_bar[1:], torch.zeros(1, device=device)), 0)

    def get_betas(self, schedule='linear'):
        if schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.T)
        elif schedule == 'cosine':
            return self.betas_for_alpha_bar(self.T,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        else :
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5,self.T) **2
    
    def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)


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
    
      
    def ddim_sample_onestep(self, model, x_t, i, batch_size):
        """
        Sample from p(x{t-1} | x_t) using the reverse process and model
        """
        t = (torch.ones(batch_size) * i).long().to(self.device)

        # predict noise using model
        eps_theta = model(x_t,t)

        # calculate x_{t-1}
        sigma_t = 0
        x_t_minus_one = (
                torch.sqrt(self.alphas_bar_prev[t] / self.alphas_bar[t])[:, None, None, None] * x_t +
                ((torch.sqrt(1 - self.alphas_bar_prev[t] - sigma_t ** 2))[:, None, None, None] - torch.sqrt(
                    (self.alphas_bar_prev[t] * (1 - self.alphas_bar[t])) / self.alphas_bar[t])[:, None, None, None]) * eps_theta 
        )

        return x_t_minus_one
    

    def ddim_sample_loop(self, model, x, t_step, batch_size, timesteps_to_save=None, verbose=True):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """ 
        """
        y is class label
        """
        if verbose:
            logging.info(f"Sampling {batch_size} new images....")
            pbar = tqdm(reversed(range(1, t_step)), position=0, total=t_step-1)
        else :
            pbar = reversed(range(1, t_step))
            
        model.eval()
        if timesteps_to_save is not None:
            intermediates = []
        with torch.no_grad():

            for i in pbar:
                # T-1, T-2, .... 0
                x = self.ddim_sample_onestep(model, x, i, batch_size)
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

