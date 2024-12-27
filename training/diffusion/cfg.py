# Classifier Free Guidance diffusion, modified from CFG-MNIST: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }



class CFG_DDPM(nn.Module):
    def __init__(
            self,
            beta1=1e-4,
            beta2=0.02,
            num_timesteps=1000,
            denoising_model=None,
            optimizer=None,
        ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.denoising_model = denoising_model.to(self.device)
        self.num_timesteps = num_timesteps

        for k, v in ddpm_schedules(beta1, beta2, num_timesteps).items():
            self.register_buffer(k, v.to(self.device))

        self.drop_prob = 0.1
        self.optimizer = optimizer
        self.loss = nn.MSELoss()
    
    def add_noise(self, x, t=None, noise=None):
        """
        x: input image
        """
        x = x.to(self.device)

        if noise is not None:
            assert noise.shape == x.shape, "Noise shape must match input shape"
            noise = noise.to(self.device)
        else:
            noise = torch.randn_like(x).to(self.device)

        if t is not None:
            assert t.shape[0] == x.shape[0], "Batch size must match"
            t = t.to(self.device)
        else:
            t = torch.randint(0, self.num_timesteps+1, (x.shape[0],)).to(self.device)
        
        x_t = extract(self.sqrtab, t, x.shape) * x + \
            extract(self.sqrtmab, t, noise.shape) * noise
        return x_t, t, noise
    
    def train_step(self, ds):
        """
        Train the denoising model.
        """
        self.denoising_model.train()
        total_loss = 0
        for x, label in tqdm.tqdm(ds):
            self.optimizer.zero_grad()
            x, label = x.to(self.device), label.to(self.device)
            x_t, t, noise = self.add_noise(x)
            
            # apply dropout and convert to one-hot
            dropout = torch.ones_like(label).float()
            dropout = F.dropout(dropout, p=self.drop_prob, training=True)
            label = F.one_hot(label, num_classes=10).float()
            label = label * dropout.unsqueeze(1)

            # normalize t
            t = t.float() / self.num_timesteps

            pred_noise = self.denoising_model(x_t, t, label)
            loss = self.loss(pred_noise, noise)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        return total_loss

    @torch.no_grad()
    def sample(self, out_shape, label, guidance=0.0):
        self.denoising_model.eval()
        x_t = torch.randn(out_shape).to(self.device)

        label = F.one_hot(label, num_classes=10).float().to(self.device)
        label = torch.cat([label, torch.zeros_like(label)], dim=0) # no guidance
        
        x_t_res = None
        x_t_store = []
        for t in range(self.num_timesteps, 0, -1):
            t_tensor = torch.tensor(t).float() / self.num_timesteps
            t_tensor = t_tensor.to(self.device).repeat(x_t.shape[0] * 2)
            x_t = torch.cat([x_t, x_t], dim=0)
            pred_noise = self.denoising_model(x_t, t_tensor, label)

            cond_pred_noise = pred_noise[:x_t.shape[0] // 2]
            uncond_pred_noise = pred_noise[x_t.shape[0] // 2:]
            x_t = x_t[:x_t.shape[0] // 2]

            final_noise = (1 + guidance) * cond_pred_noise - guidance * uncond_pred_noise
            x_t = (
                (self.oneover_sqrta[t]) * 
                (x_t - self.mab_over_sqrtmab[t] * final_noise)
                + self.sqrt_beta_t[t] * torch.randn_like(x_t).to(self.device)
            )
            if t%20==0 or t==self.num_timesteps or t<8:
                x_t_res = x_t.detach().cpu().numpy()
                x_t_store.append(x_t_res)
        
        x_t_store = np.array(x_t_store)
        return x_t, x_t_store
