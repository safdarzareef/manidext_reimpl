
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputProcess_HandPose(nn.Module): 
    # Input embedding for bimanual hand poses (B x L x 2 x 99) -> (B x L x 2 x 51) - > (B x L x 128)
    def __init__(self, num_joints=15, latent_dim=128):
        super().__init__()
        self.input_dim = (num_joints + 1) * 6 + 3 # 6D Rotation Vectors for each joint + Global Orient + Trans
        self.pose_latent_dim = (num_joints + 1) * 3
        self.latent_dim = latent_dim
        self.transEmbedding = nn.Linear(3, 3) # Global Translation Embedding
        self.poseEmbedding = nn.Linear(self.input_dim - 3, self.pose_latent_dim) # Pose Embedding
        self.multiHandEmbedding = nn.Linear((self.pose_latent_dim + 3) * 2, self.latent_dim) # Single hand embeddings to multi-hand

    def forward(self, x):
        if x.ndim == 4:
            bs, seqlen, _, _ = x.shape
            x = x.view(bs, seqlen, -1)
        
        bs, seqlen, inp_dim = x.shape
        assert inp_dim == (2 * self.input_dim), f"Input dimension mismatch. Expected {(2 * self.input_dim)}, got {inp_dim}"
        x = x.permute(1, 0, 2) # [seqlen, bs, inp_dim]
        x_hand1, x_hand2 = x[:, :, :self.input_dim], x[:, :, self.input_dim:] # [seqlen, bs, input_dim]

        trans1 = x_hand1[:, :, :3] # Global Translation  [seqlen, bs, 3]
        trans_emb1 = self.transEmbedding(trans1) # [seqlen, bs, 3]
        pose1 = x_hand1[:, :, 3:] # Hand Poses [seqlen, bs, (num_joints + 1) * 6]
        poscorrMapEmb1 = self.poseEmbedding(pose1) # [seqlen, bs, (num_joints + 1) * 3]
        hand1_emb = torch.cat((trans_emb1, poscorrMapEmb1), axis=-1) # [seqlen, bs, 3 + (num_joints + 1) * 3]

        trans2 = x_hand2[:, :, :3] # Global Translation  [seqlen, bs, 3]
        trans_emb2 = self.transEmbedding(trans2) # [seqlen, bs, 3]
        pose2 = x_hand2[:, :, 3:] # Hand Poses [seqlen, bs, (num_joints + 1) * 6]
        poscorrMapEmb2 = self.poseEmbedding(pose2) # [seqlen, bs, (num_joints + 1) * 3]
        hand2_emb = torch.cat((trans_emb2, poscorrMapEmb2), axis=-1) # [seqlen, bs, 3 + (num_joints + 1) * 3]

        multi_hand_emb = self.multiHandEmbedding(torch.cat((hand1_emb, hand2_emb), axis=-1)) # [seqlen, bs, latent_dim]
        return multi_hand_emb



class OutputProcess_HandPose(nn.Module): 
    # Output processing for bimanual hand pose embeddings (B x L x 128) -> (B x L x 2 x 51) -> (B x L x 2 x 99)
    def __init__(self, num_joints=15, latent_dim=128):
        super().__init__()
        self.input_dim = (num_joints + 1) * 6 + 3 # 6D Rotation Vectors for each joint + Global Orient + Trans
        self.pose_latent_dim = (num_joints + 1) * 3
        self.latent_dim = latent_dim

        self.multiHandFinal = nn.Linear(self.latent_dim, (self.pose_latent_dim + 3) * 2) # Multi-hand to single hand embeddings
        self.transFinal = nn.Linear(3, 3) # Global Translation Embedding to final translation
        self.poseFinal = nn.Linear(self.pose_latent_dim, self.input_dim - 3) # Pose Embedding to final 6D pose

    def forward(self, out):
        seqlen, bs, latent_dim = out.shape
        assert latent_dim == self.latent_dim, f"Latent dimension mismatch. Expected {self.latent_dim}, got {latent_dim}"
        out = self.multiHandFinal(out)
        out = out.view(seqlen, bs, 2, -1)

        out_hand1, out_hand2 = out[:, :, 0], out[:, :, 1] # [seqlen, bs, (pose_latent_dim + 3)]

        out_trans1 = out_hand1[:, :, :3] # Global Translation  [seqlen, bs, 3]
        out_trans1 = self.transFinal(out_trans1) # [seqlen, bs, 3]
        out_pose1 = out_hand1[:, :, 3:] # Hand Poses [seqlen, bs, (num_joints + 1) * 3]
        out_pose1 = self.poseFinal(out_pose1) # [seqlen, bs, (num_joints + 1) * 6]
        out_hand1 = torch.cat((out_trans1, out_pose1), axis=-1) # [seqlen, bs, input_dim]

        out_trans2 = out_hand2[:, :, :3] # Global Translation  [seqlen, bs, 3]
        out_trans2 = self.transFinal(out_trans2) # [seqlen, bs, 3]
        out_pose2 = out_hand2[:, :, 3:] # Hand Poses [seqlen, bs, (num_joints + 1) * 3]
        out_pose2 = self.poseFinal(out_pose2) # [seqlen, bs, (num_joints + 1) * 6]
        out_hand2 = torch.cat((out_trans2, out_pose2), axis=-1) # [seqlen, bs, input_dim]

        out = torch.cat((out_hand1, out_hand2), axis=-1) # [seqlen, bs, 2 * input_dim]
        out = out.permute(1, 0, 2).view(bs, seqlen, 2, -1)
        return out
    

class EmbedObjBPSMotion(nn.Module):
    # Embed Object BPS, Object starting pose and motion linear and angular velocities.
    def __init__(self, bps_dim=512, seqlen=120, latent_dim=128, use_artic_vel=True):
        super().__init__()
        self.bps_dim = bps_dim
        self.use_artic_vel = use_artic_vel
        self.seqlen = seqlen
        self.latent_dim = latent_dim
        
        self.bpsEmb = nn.Linear(self.bps_dim * 3, self.bps_dim)
        # the velocities are seqlen + 1 since initial frame pose is also used.
        self.transVelEmb = nn.Linear((seqlen + 1) * 3, latent_dim)
        self.angularVelEmb = nn.Linear((seqlen + 1)  * 6, latent_dim) # from 6d rot
        self.articVelEmb = nn.Linear((seqlen + 1) * 6, latent_dim) # from 6d rot
        self.finalEmb = nn.Linear(self.bps_dim + latent_dim * 3, latent_dim)
    
    def forward(self, bps, trans_v, angular_v, artic_v):
        bs, seqlen, _ = trans_v.shape
        bs, bps_dim, _ = bps.shape
        assert bps_dim == self.bps_dim, f"Input dimension mismatch. Expected {self.bps_dim=}, got {bps_dim=}"
        assert seqlen == (self.seqlen + 1), f"Input dimension mismatch. Expected {(self.seqlen + 1)=}, got {seqlen=}"
        trans_v = trans_v.view(bs, -1)
        angular_v = angular_v.view(bs, -1)
        artic_v = artic_v.view(bs, -1)

        bps = bps.view(bs, -1) # [bs, bps_dim, 3] -> [bs, bps_dim * 3]
        bps_emb = self.bpsEmb(bps) # [bs, 128]

        trans_vel_emb  = self.transVelEmb(trans_v)
        angular_vel_emb = self.angularVelEmb(angular_v)
        if self.use_artic_vel:
            artic_vel_emb = self.articVelEmb(artic_v)
        else:
            artic_vel_emb = torch.zeros(bs, self.latent_dim).to(trans_v.device)
        
        emb = torch.cat((bps_emb, trans_vel_emb, angular_vel_emb, artic_vel_emb), axis=-1) # [bs, latent_dim * 4]
        emb = self.finalEmb(emb) # [bs, latent_dim]
        return emb
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class MDM(nn.Module):
    def __init__(self, ff_size=1024, num_layers=4, num_heads=4, 
                 dropout=0.1, activation="gelu", seqlen=120,
                 num_joints=15, bps_dim=512, use_artic_vel=True, 
                 **kwargs):
        super().__init__()
        self.num_joints = num_joints
        self.bps_dim = bps_dim
        self.seqlen = seqlen
        self.use_artic_vel = use_artic_vel

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.latent_dim = 128
        
        self.cond_mode = kwargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kwargs.get('cond_mask_prob', 0.)
        self.input_process = InputProcess_HandPose(num_joints=num_joints, latent_dim=self.latent_dim)
        self.output_process = OutputProcess_HandPose(num_joints=num_joints, latent_dim=self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_obj_motion = EmbedObjBPSMotion(
            bps_dim=self.bps_dim, 
            use_artic_vel=self.use_artic_vel,
            latent_dim=self.latent_dim,
            seqlen=self.seqlen,
            )

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, num_joints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if y is not None:
            force_mask = y.get('uncond', False)     
            obj_emb = self.embed_obj_motion(**y['obj_motion'])
            emb += self.mask_cond(obj_emb, force_mask=force_mask)

        x = self.input_process(x)
        x = torch.cat((emb, x), axis=0)
        xseq = self.sequence_pos_encoder(x)
        output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:]
        output = self.output_process(output)
        return output


if __name__ == "__main__":
    inp_pose = torch.randn(4, 120, 2, 99)
    inp_obj_motion_emb = {
        'bps': torch.randn(4, 512, 3),
        'trans_v': torch.randn(4, 121, 3),
        'angular_v': torch.randn(4, 121, 6),
        'artic_v': torch.randn(4, 121, 6),
    }
    timesteps = torch.randint(0, 500, (4,))
    mdm = MDM()
    out = mdm(inp_pose, timesteps, y={'obj_motion': inp_obj_motion_emb})
    print(out.shape)