# Training and optimizing the MANO Continuous Surface Embedding
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tqdm

import torch
import torch.nn.functional as F

from utils.mano_helper import MANOWrapper


class MANOSurfaceEmb:
    def __init__(self, is_rhand, emb_dim=10, device="cuda"):
        flat_mano = MANOWrapper(
            is_rhand=is_rhand,
            )
        self.faces = flat_mano.model.faces
        
        self.default_vertices = flat_mano.get_mano_output().vertices[0].detach().cpu() # (778, 3)
        self.emb_dim = emb_dim
        self.device = device
        hand_str = "right" if is_rhand else "left"
        self.save_path = f"./models/cse/MANO_{hand_str}_cse_d_{self.emb_dim}.pt"
        self.load_embeddings()
        
    def compute_phi(self, distance_matrix, is_embedding=True, sigma=0.005):
        if is_embedding:
            # phi_emd = exp(-||Ei - Ej||)
            return torch.exp(-distance_matrix)
        else:
            # phi_gt = exp(-G^2 / 2σ^2)
            return torch.exp(-distance_matrix**2 / (2 * sigma**2))
        
    def load_embeddings(self):
        try:
            self.emb = torch.load(self.save_path, map_location=self.device)
            self.emb.requires_grad = True
        except:
            print(f"Embeddings not found at {self.save_path}, running and saving it")
            self.optimize_embeddings()

    def optimize_embeddings(self, num_iters=10000):
        G_dist_matrix = torch.cdist(self.default_vertices, self.default_vertices, p=2).to(self.device)
        phi_gt = self.compute_phi(G_dist_matrix, is_embedding=False)

        # Initialize embeddings
        self.emb = torch.nn.Parameter(torch.randn(778, self.emb_dim).to(self.device))
        self.emb.requires_grad = True
        
        optimizer = torch.optim.Adam([self.emb], lr=0.01)
        
        with tqdm.tqdm(range(num_iters), desc="Optimizing embeddings") as pbar:
            for it in pbar:
                optimizer.zero_grad()
                emb_dist_matrix = torch.cdist(self.emb, self.emb, p=2)
                phi_emd = self.compute_phi(emb_dist_matrix, is_embedding=True)
                
                loss = F.binary_cross_entropy(phi_emd, phi_gt)
                loss.backward()
                optimizer.step()
                
                if it % 100 == 0:
                    pbar.set_description(f"Iter {it} Loss {loss.item():.6f}")
        self.save_embeddings()

    def save_embeddings(self):
        print(f"Saving embeddings to {self.save_path}")
        torch.save(self.emb.detach().cpu(), self.save_path)

