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

import smplx

from training.diffusion.baseline_mdm import MDM

import wandb

from data.canon_seq import uncanon_seq_batch


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


class MANO_Loss(nn.Module):
    def __init__(self, w_data=0.4, w_joints=0.3, w_vel=0.3):
        super().__init__()
        self.w_data = w_data
        self.w_joints = w_joints
        self.w_vel = w_vel
        self.mse_loss = nn.MSELoss()

        self.rh_mano = smplx.create(
            model_path="./models",
            model_type="mano",
            is_rhand=True,
            use_pca=False, # no pca only use rotation
            flat_hand_mean=False, # initialize hand pose to flat hand (pose rot zero = flat hand)
        ).to("cuda")

        self.lh_mano = smplx.create(
            model_path="./models",
            model_type="mano",
            is_rhand=False,
            use_pca=False, # no pca only use rotation
            flat_hand_mean=False, # initialize hand pose to flat hand (pose rot zero = flat hand)
        ).to("cuda")

    def get_joints(self, pose, beta, global_orient, transl, is_rhand=False):
        if is_rhand:
            mano = self.rh_mano
        else:
            mano = self.lh_mano
        bs, seqlen = pose.shape[:2]
        beta = beta.unsqueeze(1).expand(-1, seqlen, -1)
        beta = beta.reshape(bs * seqlen, -1)
        pose = pose.view(bs * seqlen, -1)
        transl = transl.view(bs * seqlen, -1)
        global_orient = global_orient.view(bs * seqlen, -1)

        jts =  mano(hand_pose=pose, betas=beta, transl=transl, global_orient=global_orient).joints
        jts = jts.view(bs, seqlen, -1, 3)
        return jts
    
    def forward(self, pred_hand_pose, target_hand_pose, **kwargs):
        data_loss = self.mse_loss(pred_hand_pose, target_hand_pose)
        betas = kwargs.get("betas")
        rh_betas, lh_betas = betas[:, 0, :], betas[:, 1, :]
        _, decoded_mano_gt = uncanon_seq_batch(
            target_hand_pose, 
            trans_v=kwargs.get("trans_v"), 
            angular_v=kwargs.get("angular_v"), 
            artic_v=kwargs.get("artic_v")
            )
        _, decoded_mano_pred = uncanon_seq_batch(
            pred_hand_pose,
            trans_v=kwargs.get("trans_v"),
            angular_v=kwargs.get("angular_v"),
            artic_v=kwargs.get("artic_v")
            )
        joints_loss = 0

        rh_pose_gt, rh_trans_gt, rh_rot_gt = decoded_mano_gt["right"]["pose"], decoded_mano_gt["right"]["trans"], decoded_mano_gt["right"]["rot"]
        lh_pose_gt, lh_trans_gt, lh_rot_gt = decoded_mano_gt["left"]["pose"], decoded_mano_gt["left"]["trans"], decoded_mano_gt["left"]["rot"]
        rh_pose_pred, rh_trans_pred, rh_rot_pred = decoded_mano_pred["right"]["pose"], decoded_mano_pred["right"]["trans"], decoded_mano_pred["right"]["rot"]
        lh_pose_pred, lh_trans_pred, lh_rot_pred = decoded_mano_pred["left"]["pose"], decoded_mano_pred["left"]["trans"], decoded_mano_pred["left"]["rot"]

        rh_joints_gt = self.get_joints(
            pose=rh_pose_gt,
            beta=rh_betas,
            global_orient=rh_rot_gt,
            transl=rh_trans_gt,
            is_rhand=True
        )
        lh_joints_gt = self.get_joints(
            pose=lh_pose_gt,
            beta=lh_betas,
            global_orient=lh_rot_gt,
            transl=lh_trans_gt,
            is_rhand=False
        )
        rh_joints_pred = self.get_joints(
            pose=rh_pose_pred,
            beta=rh_betas,
            global_orient=rh_rot_pred,
            transl=rh_trans_pred,
            is_rhand=True
        )
        lh_joints_pred = self.get_joints(
            pose=lh_pose_pred,
            beta=lh_betas,
            global_orient=lh_rot_pred,
            transl=lh_trans_pred,
            is_rhand=False
        )

        joints_loss += self.mse_loss(rh_joints_pred, rh_joints_gt)
        joints_loss += self.mse_loss(lh_joints_pred, lh_joints_gt)

        vel_loss = 0
        vel_loss += self.mse_loss(rh_joints_pred[:, 1:] - rh_joints_pred[:, :-1], rh_joints_gt[:, 1:] - rh_joints_gt[:, :-1])
        vel_loss += self.mse_loss(lh_joints_pred[:, 1:] - lh_joints_pred[:, :-1], lh_joints_gt[:, 1:] - lh_joints_gt[:, :-1])

        total_loss = self.w_data * data_loss + self.w_joints * joints_loss + self.w_vel * vel_loss

        return {
            "total_loss": total_loss,
            "data_loss": data_loss,
            "joints_loss": joints_loss,
            "vel_loss": vel_loss,
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
        self.loss = MANO_Loss()

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
        final_loss_dict = {}
        for data in tqdm.tqdm(ds):
            self.optimizer.zero_grad()

            x = data["hand_pose"].to(self.device)
            x_orig = x.clone()
            obj_motion = {
                "bps": data["bps"].to(self.device),
                "trans_v": data["trans_v"].to(self.device),
                "angular_v": data["angular_v"].to(self.device),
                "artic_v": data["artic_v"].to(self.device),
            }
            betas = data["betas"].to(self.device)
            y = {"obj_motion": obj_motion}
            x_t, t, noise = self.add_noise(x)

            pred = self.denoising_model(x_t, t, y)
            loss_dict = self.loss(
                pred_hand_pose=pred,
                target_hand_pose=x_orig,
                betas=betas, 
                trans_v=obj_motion["trans_v"], 
                angular_v=obj_motion["angular_v"], 
                artic_v=obj_motion["artic_v"]
                )
            
            loss = loss_dict["total_loss"]
            loss.backward()
            for k, v in loss_dict.items():
                k = f"train_{k}"
                if k not in final_loss_dict:
                    final_loss_dict[k] = 0
                final_loss_dict[k] += v.item()

            self.optimizer.step()
        return final_loss_dict
    
    @torch.no_grad()
    def val_step(self, ds):
        """
        Validate the denoising model.
        """
        self.denoising_model.eval()
        final_loss_dict = {}
        for data in tqdm.tqdm(ds):
            x = data["hand_pose"].to(self.device)
            x_orig = x.clone()
            obj_motion = {
                "bps": data["bps"].to(self.device),
                "trans_v": data["trans_v"].to(self.device),
                "angular_v": data["angular_v"].to(self.device),
                "artic_v": data["artic_v"].to(self.device),
            }
            betas = data["betas"].to(self.device)
            y = {"obj_motion": obj_motion}
            x_t, t, noise = self.add_noise(x)

            pred = self.denoising_model(x_t, t, y)
            loss_dict = self.loss(
                pred_hand_pose=pred,
                target_hand_pose=x_orig,
                betas=betas, 
                trans_v=obj_motion["trans_v"], 
                angular_v=obj_motion["angular_v"], 
                artic_v=obj_motion["artic_v"]
            )
            for k, v in loss_dict.items():
                k = f"val_{k}"
                if k not in final_loss_dict:
                    final_loss_dict[k] = 0
                final_loss_dict[k] += v.item()
        return final_loss_dict

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
            # cond_pred_noise = pred_noise[:x_t.shape[0] // 2]
            # uncond_pred_noise = pred_noise[x_t.shape[0] // 2:]
            # x_t = x_t[:x_t.shape[0] // 2]

            # final_noise = (1 + guidance) * cond_pred_noise - guidance * uncond_pred_noise
            
            # if t%20==0 or t==self.num_timesteps or t<8:
            #     x_t_res = x_t.detach().cpu().numpy()
            #     x_t_store.append(x_t_res)
        
        # x_t_store = np.array(x_t_store)
        return x_t


def train(args):
    wandb.require("core")
    wandb.login()
    wandb.init(
        project="manidext_baseline_mdm",
        name=f"run_{2}",
        config=args,
    )

    ds_train = ArcticDataloader(
        data_root=args.arctic_path,
        split="train",
        bps_dim=args.bps_dim,
        cse_dim=args.mano_cse_dim,
        return_fixed_length=args.seqlen,
        return_items=["bps", "canon_seq"]
    )
    ds_val = ArcticDataloader(
        data_root=args.arctic_path,
        split="val",
        bps_dim=args.bps_dim,
        cse_dim=args.mano_cse_dim,
        return_fixed_length=args.seqlen,
        return_items=["bps", "canon_seq"]
    )

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    denoising_model = MDM(
        seqlen=args.seqlen,
        use_artic_vel=True,
        bps_dim=args.bps_dim,
    )

    cfg = CFG_DDPM(
        beta1=1e-4,
        beta2=0.02,
        num_timesteps=args.num_timesteps,
        denoising_model=denoising_model,
    )

    load_iter = args.load_iter
    if os.path.exists(f"./models/baseline/iter_{load_iter:06d}.pth"):
        print("Loading pre-trained model ...")
        cfg.load_state_dict(torch.load(f"./models/baseline/iter_{load_iter:06d}.pth"))
        start_ep = load_iter // len(dl_train)
    else:
        start_ep = 0

    curr_lr = 2e-4
    cfg.optimizer = torch.optim.Adam(cfg.denoising_model.parameters(), lr=curr_lr)

    os.makedirs("./models/baseline/", exist_ok=True)

    num_epochs = args.num_iters // len(dl_train)

    for epoch in range(start_ep, num_epochs):
        curr_iter = (epoch+1) * len(dl_train)
        print(f"Running trainstep to iter {curr_iter}/{args.num_iters}  ...")
        train_loss_dict = cfg.train_step(dl_train)
        print(f"Running valstep to iter {curr_iter}/{args.num_iters}  ...")
        val_loss_dict = cfg.val_step(dl_val)
        print(f"Epoch: {epoch}, Train Loss: {train_loss_dict['train_total_loss']}, Val Loss: {val_loss_dict['val_total_loss']}")

        log_dict = {"iter": curr_iter, "lr": curr_lr}
        log_dict.update(train_loss_dict)
        log_dict.update(val_loss_dict)
        wandb.log(log_dict)

        if epoch % 20 == 0 and epoch > 0:
            print(f"Saving model at ./models/baseline/iter_{curr_iter:06d}.pth ...")
            torch.save(cfg.state_dict(), f"./models/baseline/iter_{curr_iter:06d}.pth")
    wandb.finish()


def sample(args):
    denoising_model = MDM(
        seqlen=args.seqlen,
        use_artic_vel=True,
        bps_dim=args.bps_dim,
    )

    cfg = CFG_DDPM(
        beta1=1e-4,
        beta2=0.02,
        num_timesteps=args.num_timesteps,
        denoising_model=denoising_model,
    )

    load_iter = args.load_iter
    if os.path.exists(f"./models/baseline/iter_{load_iter:06d}.pth"):
        print("Loading pre-trained model ...")
        cfg.load_state_dict(torch.load(f"./models/baseline/iter_{load_iter:06d}.pth"))
    else:
        raise FileNotFoundError("Model not found.")
    
    ds_val = ArcticDataloader(
        data_root=args.arctic_path,
        split="val",
        bps_dim=args.bps_dim,
        cse_dim=args.mano_cse_dim,
        return_fixed_length=args.seqlen,
        return_items=["bps", "canon_seq", "bps_vis"]
    )
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)

    data = next(iter(dl_val))
    obj_motion = {
                "bps": data["bps"].to(cfg.device),
                "trans_v": data["trans_v"].to(cfg.device),
                "angular_v": data["angular_v"].to(cfg.device),
                "artic_v": data["artic_v"].to(cfg.device),
            }
    betas = data["betas"].to(cfg.device)
    target_hand_pose = data["hand_pose"].to(cfg.device)
    pred_hand_pose = cfg.sample(target_hand_pose.shape, y={"obj_motion": obj_motion})

    rh_betas, lh_betas = betas[:, 0, :], betas[:, 1, :]
    decoded_obj_seq_gt, decoded_mano_gt = uncanon_seq_batch(
        target_hand_pose, 
        trans_v=obj_motion.get("trans_v"), 
        angular_v=obj_motion.get("angular_v"), 
        artic_v=obj_motion.get("artic_v")
        )
    decoded_obj_seq_pred, decoded_mano_pred = uncanon_seq_batch(
        pred_hand_pose,
        trans_v=obj_motion.get("trans_v"),
        angular_v=obj_motion.get("angular_v"),
        artic_v=obj_motion.get("artic_v")
        )

    obj_motion = {
        "bps": data["bps"].to("cpu"),
        "trans_v": data["trans_v"].to("cpu"),
        "angular_v": data["angular_v"].to("cpu"),
        "artic_v": data["artic_v"].to("cpu"),
    }
    sampled_res = {
        "pred": pred_hand_pose.detach().cpu(),
        "gt": target_hand_pose.detach().cpu(),
        "obj_motion": obj_motion,
        "bps_vis": data["bps_vis"],
        "betas": betas.to("cpu"),
        "pred_mano": decoded_mano_pred,
        "gt_mano": decoded_mano_gt,
        "pred_obj": decoded_obj_seq_pred.cpu(),
        "gt_obj": decoded_obj_seq_gt.cpu(),
    }
    import pickle
    os.makedirs("./models/samples/baseline/", exist_ok=True)
    pickle.dump(sampled_res, open(f"./models/samples/baseline/sample_{load_iter:06d}.pkl", "wb"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic_path", type=str,
                        default="/home/zareef/Datasets/arctic/unpack/arctic_data/",
                        # required=True, 
                        help="Path to unpacked arctic dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--seqlen", type=int, default=120, help="Sequence length for clipping data")
    parser.add_argument("--num_iters", type=int, default=300000, help="Number of iters to train")
    parser.add_argument("--num_timesteps", type=int, default=500, help="Number of timesteps for DDPM")
    parser.add_argument("--bps_dim", type=int, default=512, help="Basis Point Set dimension")
    parser.add_argument("--mano_cse_dim", type=int, default=16, help="MANO Continuous Surface Embedding dimension")
    parser.add_argument("--load_iter", type=int, default=21610, help="Load model from iteration")
    parser.add_argument("--sample_mode", action="store_true", default=False, help="Sample from trained model")
    args = parser.parse_args()
    
    if not args.sample_mode:
        train(args)
    else:
        sample(args)