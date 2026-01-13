import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))



class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    def forward(self, x_0, labels):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction="none")
        return loss



class DynamicGuidanceDiffusionSampler(nn.Module):
    def __init__(
        self,
        model,
        beta_1,
        beta_T,
        T,
        gamma: float = 1.0,   
        zeta: float = 0.5,    
        mu: float = 0.1,      
        c: float = 1.0,       
        eta: float = 0.0,     
        sigma_e: float = 1.0 
    ):
        super().__init__()
        self.model = model
        self.T = T

        self.gamma = gamma
        self.zeta = zeta
        self.mu = mu
        self.c = c
        self.eta = eta
        self.sigma_e = sigma_e

 
        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_bar", alphas)
        self.register_buffer("alphas_bar_total", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar)
        )

    def forward(
        self,
        x_T,
        labels,
        y,
        A_op=None,
        AT_op=None,
        return_intermediates: bool = False,
    ):

        if A_op is None:
            A_op = lambda x: x
        if AT_op is None:
            AT_op = lambda x: x

        device = x_T.device
        x_t = x_T
        intermediates = []

        B = x_T.size(0)

   
        use_guidance = self.sigma_e > 0

        for time_step in reversed(range(self.T)):
            t = torch.full((B,), time_step, dtype=torch.long, device=device)

      
            eps_theta = self.model(x_t, t, labels) 

            sqrt_ab_t = extract(self.sqrt_alphas_bar, t, x_t.shape)
            sqrt_1mab_t = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)

       
            x0_t = (x_t - sqrt_1mab_t * eps_theta) / (sqrt_ab_t + 1e-8)

            if use_guidance:
                alpha_bar_t = extract(self.alphas_bar_total, t, x_t.shape)
                delta_t = alpha_bar_t ** self.gamma
                w_t = delta_t
            else:
                delta_t = torch.zeros_like(x0_t[:, :1, ...]) 
                w_t = torch.ones_like(x0_t[:, :1, ...])



            Ax0 = A_op(x0_t)
            r = Ax0 - y


            gBP = r / (1.0 + self.eta)
            gLS = self.c * r

            g_delta = (1.0 - delta_t) * gBP + delta_t * gLS


            x_hat_prev = x0_t - self.mu * g_delta


            eps_hat = (x_t - sqrt_ab_t * x_hat_prev) / (sqrt_1mab_t + 1e-8)

            if time_step > 0:
                eps_rand = torch.randn_like(x_t)
            else:
                eps_rand = torch.zeros_like(x_t)

            if time_step > 0:
                t_prev = torch.full((B,), time_step - 1, dtype=torch.long, device=device)
            else:
                t_prev = t

            sqrt_ab_prev = extract(self.sqrt_alphas_bar, t_prev, x_t.shape)
            sqrt_1mab_prev = torch.sqrt(1.0 - extract(self.alphas_bar_total, t_prev, x_t.shape))


            mix_noise = (
                w_t * torch.sqrt(torch.tensor(1.0 - self.zeta, device=device)) * eps_hat
                + torch.sqrt(torch.tensor(self.zeta, device=device)) * eps_rand
            )

            x_t = sqrt_ab_prev * x_hat_prev + sqrt_1mab_prev * mix_noise

            if return_intermediates:
                intermediates.append((time_step, x_t.clone()))

        x_0 = torch.clip(x_t, -1.0, 1.0)
        if return_intermediates:
            return x_0, intermediates
        return x_0
