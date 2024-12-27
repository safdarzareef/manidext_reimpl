# Classifier Free Guidance diffusion, modified from CFG-MNIST: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.arctic_dataloader import ArcticDataloader
from torch.utils.data import DataLoader

from training.diffusion.modified_mdm import MDM

import wandb


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


class Stage1_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        target_contact = target[:, :, :, :, 0].clamp(0, 1)
        target_emb = target[:, :, :, :, 1:][target_contact >= 0.5]

        pred_contact = F.sigmoid(pred[:, :, :, :, 0])
        pred_emb = pred[:, :, :, :, 1:][target_contact >= 0.5]

        loss = self.bce_loss(pred_contact, target_contact)
        loss += self.mse_loss(pred_emb, target_emb)
        return loss


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
        # self.loss = nn.MSELoss()
        self.loss = Stage1_Loss()

    def change_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
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
        for data in tqdm.tqdm(ds):
            self.optimizer.zero_grad()

            x = data["ce_seq"].to(self.device)
            x_orig = x.clone()
            obj_motion = {
                "bps": data["bps"].to(self.device),
                "trans_v": data["trans_v"].to(self.device),
                "angular_v": data["angular_v"].to(self.device),
                "artic_v": data["artic_v"].to(self.device),
            }
            y = {"obj_motion": obj_motion}
            x_t, t, noise = self.add_noise(x)

            pred = self.denoising_model(x_t, t, y)
            # loss = self.loss(pred_noise, noise)
            loss = self.loss(pred, x_orig)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        return total_loss
    
    @torch.no_grad()
    def val_step(self, ds):
        """
        Validate the denoising model.
        """
        self.denoising_model.eval()
        total_loss = 0
        for data in tqdm.tqdm(ds):
            x = data["ce_seq"].to(self.device)
            x_orig = x.clone()
            obj_motion = {
                "bps": data["bps"].to(self.device),
                "trans_v": data["trans_v"].to(self.device),
                "angular_v": data["angular_v"].to(self.device),
                "artic_v": data["artic_v"].to(self.device),
            }
            y = {"obj_motion": obj_motion}
            x_t, t, noise = self.add_noise(x)

            pred = self.denoising_model(x_t, t, y)
            loss = self.loss(pred, x_orig)
            total_loss += loss.item()
        return total_loss

    @torch.no_grad()
    def sample(self, out_shape, y, guidance=0.0):
        self.denoising_model.eval()
        x_t = torch.randn(out_shape).to(self.device)

        x_t_res = None
        # x_t_store = []
        for t in range(self.num_timesteps, 0, -1):
            t_tensor = torch.tensor(t).to(self.device)
            t_tensor = t_tensor.to(self.device).repeat(x_t.shape[0])
            # x_t = torch.cat([x_t, x_t], dim=0)
            x_t = self.denoising_model(x_t, t_tensor, y)

            # # cond_pred_noise = pred_noise[:x_t.shape[0] // 2]
            # # uncond_pred_noise = pred_noise[x_t.shape[0] // 2:]
            # # x_t = x_t[:x_t.shape[0] // 2]

            # # final_noise = (1 + guidance) * cond_pred_noise - guidance * uncond_pred_noise
            # x_t = (
            #     (self.oneover_sqrta[t]) * 
            #     (x_t - self.mab_over_sqrtmab[t] * pred_noise)
            #     + self.sqrt_beta_t[t] * torch.randn_like(x_t).to(self.device)
            # )
            # if t%20==0 or t==self.num_timesteps or t<8:
            #     x_t_res = x_t.detach().cpu().numpy()
            #     x_t_store.append(x_t_res)
        
        # x_t_store = np.array(x_t_store)
        return x_t
    


def train(args):
    wandb.require("core")
    wandb.login()
    wandb.init(
        project="manidext_stage1",
        name=f"run_{1}",
        config=args,
    )

    ds_train = ArcticDataloader(
        data_root=args.arctic_path,
        split="train",
        bps_dim=args.bps_dim,
        cse_dim=args.mano_cse_dim,
        return_fixed_length=args.seqlen,
        return_items=["bps", "ce_seq", "canon_seq"]
    )
    ds_val = ArcticDataloader(
        data_root=args.arctic_path,
        split="val",
        bps_dim=args.bps_dim,
        cse_dim=args.mano_cse_dim,
        return_fixed_length=args.seqlen,
        return_items=["bps", "ce_seq", "canon_seq"]
    )

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    denoising_model = MDM(
        modeltype="ce_map",
        seqlen=args.seqlen,
        use_artic_vel=True,
        bps_dim=args.bps_dim,
        cse_dim=args.mano_cse_dim,
    )

    cfg = CFG_DDPM(
        beta1=1e-4,
        beta2=0.02,
        num_timesteps=args.num_timesteps,
        denoising_model=denoising_model,
    )

    load_iter = args.load_iter
    if os.path.exists(f"./models/stage1/manidext_stage1_iter_{load_iter:06d}.pth"):
        print("Loading pre-trained model ...")
        cfg.load_state_dict(torch.load(f"./models/stage1/manidext_stage1_iter_{load_iter:06d}.pth"))
        start_ep = load_iter // len(dl_train)
    else:
        start_ep = 0

    curr_lr = 1e-4
    cfg.optimizer = torch.optim.Adam(cfg.denoising_model.parameters(), lr=curr_lr)

    os.makedirs("./models/stage1/", exist_ok=True)

    num_epochs = args.num_iters // len(dl_train)

    for epoch in range(start_ep, num_epochs):
        curr_iter = (epoch+1) * len(dl_train)
        print(f"Running trainstep to iter {curr_iter}/{args.num_iters}  ...")
        train_loss = cfg.train_step(dl_train)
        print(f"Running valstep to iter {curr_iter}/{args.num_iters}  ...")
        val_loss = cfg.val_step(dl_val)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        wandb.log({"iter": curr_iter, "train_loss": train_loss, "val_loss": val_loss, "lr": curr_lr})

        # if epoch == 100:
        #     curr_lr = curr_lr / 5
        #     cfg.change_lr(curr_lr)
        #     print(f"Changing learning rate to {curr_lr} ...")
        # if epoch == 500:
        #     curr_lr = curr_lr / 2
        #     cfg.change_lr(curr_lr)
        #     print(f"Changing learning rate to {curr_lr} ...")

        if epoch % 20 == 0 and epoch > 0:
            print(f"Saving model at ./models/stage1/manidext_stage1_iter_{curr_iter:06d}.pth ...")
            torch.save(cfg.state_dict(), f"./models/stage1/manidext_stage1_iter_{curr_iter:06d}.pth")
    wandb.finish()


def sample(args):
    denoising_model = MDM(
        modeltype="ce_map",
        seqlen=args.seqlen,
        use_artic_vel=True,
        bps_dim=args.bps_dim,
        cse_dim=args.mano_cse_dim,
    )

    cfg = CFG_DDPM(
        beta1=1e-4,
        beta2=0.02,
        num_timesteps=args.num_timesteps,
        denoising_model=denoising_model,
    )

    load_iter = args.load_iter
    if os.path.exists(f"./models/stage1/manidext_stage1_iter_{load_iter:06d}.pth"):
        print("Loading pre-trained model ...")
        cfg.load_state_dict(torch.load(f"./models/stage1/manidext_stage1_iter_{load_iter:06d}.pth"))
    else:
        raise FileNotFoundError("Model not found.")
    
    ds_val = ArcticDataloader(
        data_root=args.arctic_path,
        split="val",
        bps_dim=args.bps_dim,
        cse_dim=args.mano_cse_dim,
        return_fixed_length=args.seqlen,
        return_items=["bps", "ce_seq", "canon_seq", "bps_vis"]
    )
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)

    data = next(iter(dl_val))
    obj_motion = {
                "bps": data["bps"].to(cfg.device),
                "trans_v": data["trans_v"].to(cfg.device),
                "angular_v": data["angular_v"].to(cfg.device),
                "artic_v": data["artic_v"].to(cfg.device),
            }
    pred = cfg.sample(data["ce_seq"].shape, y={"obj_motion": obj_motion})

    obj_motion = {
        "bps": data["bps"].to("cpu"),
        "trans_v": data["trans_v"].to("cpu"),
        "angular_v": data["angular_v"].to("cpu"),
        "artic_v": data["artic_v"].to("cpu"),
    }
    sampled_res = {
        "pred": pred.detach().cpu(),
        "gt": data["ce_seq"].detach().cpu(),
        "obj_motion": obj_motion,
        "bps_vis": data["bps_vis"],
    }
    import pickle
    os.makedirs("./models/samples/stage1/", exist_ok=True)
    pickle.dump(sampled_res, open(f"./models/samples/stage1/sample_{load_iter:06d}.pkl", "wb"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic_path", type=str,
                        default="/home/zareef/Datasets/arctic/unpack/arctic_data/",
                        # required=True, 
                        help="Path to unpacked arctic dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seqlen", type=int, default=120, help="Sequence length for clipping data")
    parser.add_argument("--num_iters", type=int, default=300000, help="Number of iters to train")
    parser.add_argument("--num_timesteps", type=int, default=500, help="Number of timesteps for DDPM")
    parser.add_argument("--bps_dim", type=int, default=512, help="Basis Point Set dimension")
    parser.add_argument("--mano_cse_dim", type=int, default=16, help="MANO Continuous Surface Embedding dimension")
    parser.add_argument("--load_iter", type=int, default=46379, help="Load model from iteration")
    parser.add_argument("--sample_mode", action="store_true", default=False, help="Sample from trained model")
    args = parser.parse_args()
    
    if not args.sample_mode:
        train(args)
    else:
        sample(args)