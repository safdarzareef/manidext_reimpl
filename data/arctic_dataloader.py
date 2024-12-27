import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import copy
import argparse

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pytorch3d
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d
from scipy.spatial.transform import Rotation as R

from bps_torch.bps import bps_torch
from training.mano_cse import MANOSurfaceEmb
from utils.artic_obj_helper import ArticulatedObject
from utils.mano_helper import MANOWrapper
from utils.utils import *


def load_mesh_pytorch3d(file_path):
    """
    Load a mesh using PyTorch3D's built-in loaders.
    Supports .obj and .ply files.
    """
    if file_path.endswith('.obj'):
        verts, faces, aux = load_obj(file_path)
        faces = faces.verts_idx
    elif file_path.endswith('.ply'):
        verts, faces = load_ply(file_path)
    else:
        raise ValueError("Unsupported file format. Use .obj or .ply")
    
    # Create a Meshes object
    mesh = Meshes(
        verts=[verts],
        faces=[faces]
    )
    return mesh



class ArcticDataloader(Dataset):
    def __init__(
            self, 
            data_root, 
            split="train",
            bps_dim=512,
            cse_dim=16,
            return_fixed_length=-1,
            return_items=["mano_seq", "obj_seq", "obj_mesh_path", "bps_vis", "ce_seq", "canon_seq"],
            ):
        self.arctic_root = data_root
        self.train_subjects = []
        self.val_subjects = []
        self.test_subjects = []
        self._return_items = set(return_items)
        self._return_fixed_length = return_fixed_length

        for dir in os.listdir(os.path.join(self.arctic_root, "data/raw_seqs")):
            if os.path.isdir(os.path.join(self.arctic_root, "data/raw_seqs", dir)):
                self.train_subjects.append(dir)
        
        self.train_subjects = sorted(self.train_subjects)
        self.test_subjects = [self.train_subjects.pop()]
        self.val_subjects = [self.train_subjects.pop()]
        print(f"{self.train_subjects=}, {self.val_subjects=}, {self.test_subjects=}")
        self.split = split

        self.sequences = {"mano_seq": [], "object_seq": [], "object_mesh_path": []}
        self.objects = set()

        self.num_bps_points = bps_dim // 2
        self.bps_radius = 0.3
        # Initialize BPS
        if os.path.exists("./models/bps.pt"):
            self.custom_bps = torch.load("./models/bps.pt")
            self.bps = bps_torch(
                bps_type="random_uniform",
                n_bps_points=self.num_bps_points,
                radius=self.bps_radius,
                custom_basis=self.custom_bps,
            )
        else:
            self.bps = bps_torch(
                bps_type="random_uniform",
                n_bps_points=self.num_bps_points,
                radius=self.bps_radius,
            )
            self.custom_bps = copy.deepcopy(self.bps.bps)
            torch.save(self.custom_bps, "./models/bps.pt")

        self.setup_seq_paths()
        self.ds_length = len(self.sequences["mano_seq"]) * 5 # since randomly slicing clips, we want to keep the length fixed
        self._preprocess_sequences()
        self.bps_cache = {}
        self.setup_bps_cache()

        self.cse_dim = cse_dim
        self.left_emb = MANOSurfaceEmb(is_rhand=False, emb_dim=cse_dim).emb.detach().cpu()
        self.right_emb = MANOSurfaceEmb(is_rhand=True, emb_dim=cse_dim).emb.detach().cpu()
        self._setup_gt_ce_map()
    
    def setup_seq_paths(self):
        if self.split == "train":
            folder_to_use = self.train_subjects
        elif self.split == "val":
            folder_to_use = self.val_subjects
        elif self.split == "test":
            folder_to_use = self.test_subjects
        else:
            raise ValueError(f"Invalid split {self.split}")
        
        for subject in folder_to_use:
            files = os.listdir(os.path.join(self.arctic_root, "data/raw_seqs", subject))
            files = [f.split(".")[0] for f in files]
            files = set(files)
            
            for fname in files:
                obj_name = fname.split("_")[0]
                mano_path = os.path.join(self.arctic_root, "data/raw_seqs", subject, f"{fname}.mano.npy")
                obj_path = os.path.join(self.arctic_root, "data/raw_seqs", subject, f"{fname}.object.npy")
                obj_mesh_path = os.path.join(self.arctic_root, f"data/meta/object_vtemplates/{obj_name}/")
                self.objects.add(obj_name)
                if os.path.exists(mano_path) and os.path.exists(obj_path) and os.path.exists(obj_mesh_path):
                    self.sequences["mano_seq"].append(mano_path)
                    self.sequences["object_seq"].append(obj_path)
                    self.sequences["object_mesh_path"].append(obj_mesh_path)

    def setup_bps_cache(self):
        self.bps_cache = {}
        os.makedirs("./data/bps_cache/", exist_ok=True)
        for obj in self.objects:
            obj_mesh_path = os.path.join(self.arctic_root, f"data/meta/object_vtemplates/{obj}/")
            obj_bps_path = os.path.join("./data/bps_cache/", f"{obj}_bps.pt")
            if os.path.exists(obj_bps_path):
                self.bps_cache[obj_mesh_path] = torch.load(obj_bps_path)
            elif os.path.exists(obj_mesh_path):
                self.bps_cache[obj_mesh_path] = self.encode_bps(obj_mesh_path, return_decoded=True)
                torch.save(self.bps_cache[obj_mesh_path], obj_bps_path)
            
    def __len__(self):
        assert(len(self.sequences["mano_seq"]) == len(self.sequences["object_seq"]) == len(self.sequences["object_mesh_path"]))
        return self.ds_length
    
    def encode_bps(self, obj_mesh_path, return_decoded=False):
        if obj_mesh_path in self.bps_cache: # using cache as bps encoding is expensive and there's low number of objects
            bps_dict = self.bps_cache[obj_mesh_path]
            if return_decoded:
                bps_dict["top_decoded"] = self.bps.decode(bps_dict["top"], custom_basis=self.custom_bps)
                bps_dict["bottom_decoded"] = self.bps.decode(bps_dict["bottom"], custom_basis=self.custom_bps)
            else:
                for k in ["top_decoded", "bottom_decoded"]:
                    if k in bps_dict:
                        del bps_dict[k]
            return bps_dict
        
        top_mesh = load_mesh_pytorch3d(os.path.join(obj_mesh_path, "top.obj"))
        bottom_mesh = load_mesh_pytorch3d(os.path.join(obj_mesh_path, "bottom.obj"))

        # mesh in mm
        top_pcd = pytorch3d.ops.sample_points_from_meshes(top_mesh, self.num_bps_points) * (0.001) 
        bottom_pcd = pytorch3d.ops.sample_points_from_meshes(bottom_mesh, self.num_bps_points) * (0.001)

        top_bps = self.bps.encode(
                            top_pcd,
                            feature_type=['dists','deltas'],
                            custom_basis=self.custom_bps
                        )['deltas']
        bottom_bps = self.bps.encode(
                            bottom_pcd,
                            feature_type=['dists','deltas'],
                            custom_basis=self.custom_bps
                        )['deltas']
        
        bps_dict = {"top": top_bps, "bottom": bottom_bps}
        if return_decoded:
            bps_dict["top_decoded"] = self.bps.decode(top_bps, custom_basis=self.custom_bps)
            bps_dict["bottom_decoded"] = self.bps.decode(bottom_bps, custom_basis=self.custom_bps)
        return bps_dict
    
    def _transform_bps(self, bps_dict, curr_obj_rot, curr_obj_trans, curr_obj_articulation):
        # Rotate and translate the BPS points
        top_bps = bps_dict["top"]
        bottom_bps = bps_dict["bottom"]
        top_decoded = bps_dict["top_decoded"]
        bottom_decoded = bps_dict["bottom_decoded"]

        if curr_obj_articulation != 0: # just skip if no articulation
            artic_rot_mtx = np.eye(4)
            artic_rot_vec = np.array([0, 0, -curr_obj_articulation]) # need to reverse angle
            artic_rot_mtx[:3, :3] = R.from_rotvec(artic_rot_vec).as_matrix()
            # self.curr_obj_mesh_top.transform(artic_rot_mtx)
            top_decoded = transform_points(top_decoded, artic_rot_mtx)

        global_rot_mtx = np.eye(4)
        # Obj rot in axis-angle format
        global_rot_mtx[:3, :3] = R.from_rotvec(curr_obj_rot).as_matrix()

        # Global translation
        global_trans_mtx = np.eye(4)
        global_trans_mtx[:3, 3] = curr_obj_trans * (0.001) # convert to m from mm

        trans_mtx = global_trans_mtx @ global_rot_mtx
        top_decoded = transform_points(top_decoded, trans_mtx)
        bottom_decoded = transform_points(bottom_decoded, trans_mtx)
        upd_bps_dict = {
            "top": top_bps,
            "bottom": bottom_bps,
            "top_decoded": top_decoded,
            "bottom_decoded": bottom_decoded
        }
        return upd_bps_dict
    
    def canonicalize_sequences(self, mano_seq, obj_seq):
        """
        Compute Velocities and Canonicalize the object and hand sequence.
        """
        rhand_seq = mano_seq["right"]
        lhand_seq = mano_seq["left"]
        num_frames = obj_seq.shape[0]
        rhand_pose, rhand_global_orient, rhand_trans = rhand_seq["pose"], rhand_seq["rot"], rhand_seq["trans"]
        lhand_pose, lhand_global_orient, lhand_trans = lhand_seq["pose"], lhand_seq["rot"], lhand_seq["trans"]
        
        obj_angular_velocities = []
        obj_linear_velocities = []
        obj_artic_velocities = []

        new_rhand = []
        new_lhand = []
    
        for i in range(num_frames):
            curr_obj_seq = obj_seq[i]
            if i == 0:
                prev_obj_seq = curr_obj_seq.copy()
            else:
                prev_obj_seq = obj_seq[i-1]

            curr_obj_artic, curr_obj_rot, curr_obj_trans = curr_obj_seq[0], curr_obj_seq[1:4], curr_obj_seq[4:]
            prev_obj_artic, prev_obj_rot, prev_obj_trans = prev_obj_seq[0], prev_obj_seq[1:4], prev_obj_seq[4:]
            curr_obj_trans, prev_obj_trans = curr_obj_trans * 0.001, prev_obj_trans * 0.001

            # Compute angular velocity
            curr_obj_rot = R.from_rotvec(curr_obj_rot).as_matrix()
            prev_obj_rot = R.from_rotvec(prev_obj_rot).as_matrix()
            angular_vel = prev_obj_rot.T @ curr_obj_rot
            angular_vel = matrix_to_rotation_6d(torch.tensor(angular_vel)).numpy()
            obj_angular_velocities.append(angular_vel.flatten())

            # Compute linear velocity
            translational_vel = (curr_obj_trans - prev_obj_trans)
            translational_vel = prev_obj_rot.T @ translational_vel
            obj_linear_velocities.append(translational_vel.flatten())

            # Compute articulation velocity
            cur_artic_rot = np.array([0, 0, -curr_obj_artic]) # need to reverse angle
            cur_artic_rot = R.from_rotvec(cur_artic_rot).as_matrix()
            prev_artic_rot = np.array([0, 0, -prev_obj_artic]) # need to reverse angle
            prev_artic_rot = R.from_rotvec(prev_artic_rot).as_matrix()
            artic_vel = prev_artic_rot.T @ cur_artic_rot
            artic_vel = matrix_to_rotation_6d(torch.tensor(artic_vel)).numpy()
            obj_artic_velocities.append(artic_vel.flatten())

            curr_rhand_pose, curr_rhand_global_orient, curr_rhand_trans = rhand_pose[i], rhand_global_orient[i], rhand_trans[i]
            curr_lhand_pose, curr_lhand_global_orient, curr_lhand_trans = lhand_pose[i], lhand_global_orient[i], lhand_trans[i]

            curr_rhand_pose = R.from_rotvec(curr_rhand_pose.reshape(-1, 3)).as_matrix()
            curr_rhand_pose = matrix_to_rotation_6d(torch.tensor(curr_rhand_pose)).numpy()
            new_rhand_trans = curr_obj_rot.T @ (curr_rhand_trans - curr_obj_trans)
            new_rhand_global_orient = curr_obj_rot.T @ R.from_rotvec(curr_rhand_global_orient.reshape(-1, 3)).as_matrix()
            new_rhand_global_orient = matrix_to_rotation_6d(torch.tensor(new_rhand_global_orient)).numpy()

            rhand_vec = np.zeros(99)
            rhand_vec[:3] = new_rhand_trans.flatten()
            rhand_vec[3:9] = new_rhand_global_orient.flatten()
            rhand_vec[9:] = curr_rhand_pose.flatten()
            new_rhand.append(rhand_vec)

            curr_lhand_pose = R.from_rotvec(curr_lhand_pose.reshape(-1, 3)).as_matrix()
            curr_lhand_pose = matrix_to_rotation_6d(torch.tensor(curr_lhand_pose)).numpy()
            new_lhand_trans = curr_obj_rot.T @ (curr_lhand_trans - curr_obj_trans)
            new_lhand_global_orient = curr_obj_rot.T @ R.from_rotvec(curr_lhand_global_orient.reshape(-1, 3)).as_matrix()
            new_lhand_global_orient = matrix_to_rotation_6d(torch.tensor(new_lhand_global_orient)).numpy()

            lhand_vec = np.zeros(99)
            lhand_vec[:3] = new_lhand_trans.flatten()
            lhand_vec[3:9] = new_lhand_global_orient.flatten()
            lhand_vec[9:] = curr_lhand_pose.flatten()
            new_lhand.append(lhand_vec)
            

        obj_angular_velocities = np.array(obj_angular_velocities)
        obj_linear_velocities = np.array(obj_linear_velocities)
        obj_artic_velocities = np.array(obj_artic_velocities)
        new_rhand = np.array(new_rhand)
        new_lhand = np.array(new_lhand)
        return {
            "obj_angular_velocities": obj_angular_velocities,
            "obj_linear_velocities": obj_linear_velocities,
            "obj_artic_velocities": obj_artic_velocities,
            "right": new_rhand,
            "left": new_lhand,
        }
    
    def _compute_gt_ce_map_from_one_seq(self, obj_mesh_path, obj_seq, mano_seq):
        """
        Compute the ground truth object contact and embedding maps from one sequence.
        """
        def emb_map_gen(obj_verts, hand_verts, hand_embs, contact_sigma=0.25):
            vert_mapping, vert_dist = find_nearest_vertices(obj_verts, hand_verts)
            vert_embs = hand_embs[vert_mapping]
            vert_dist = np.array(vert_dist)
            vert_dist = (vert_dist - np.min(vert_dist)) / (np.max(vert_dist) - np.min(vert_dist))
            
            if np.min(vert_dist) > 0.0075: # too far away so no contact
                contact_map = np.zeros_like(vert_dist)
            else:
                vert_dist = (vert_dist - np.min(vert_dist)) / (np.max(vert_dist) - np.min(vert_dist))
                
                contact_map = np.exp(-vert_dist/(2 * contact_sigma**2)) # radial basis function
                contact_map = np.clip(contact_map, 0, 1)
            return vert_mapping, vert_embs, contact_map

        num_frames = obj_seq.shape[0]
        left_hand_seq = mano_seq["left"]
        right_hand_seq = mano_seq["right"]

        left_mano = MANOWrapper(
                is_rhand=False,
                flat_hand_mean=False,
                betas=left_hand_seq['shape'],
                init_pose=left_hand_seq['pose'][0],
                init_orient=left_hand_seq['rot'][0],
                init_trans=left_hand_seq['trans'][0],
            )
        left_emb = MANOSurfaceEmb(is_rhand=False, emb_dim=20).emb.detach().cpu()

        right_mano = MANOWrapper(
                is_rhand=True,
                flat_hand_mean=False,
                betas=right_hand_seq['shape'],
                init_pose=right_hand_seq['pose'][0],
                init_orient=right_hand_seq['rot'][0],
                init_trans=right_hand_seq['trans'][0],
            )
        right_emb = MANOSurfaceEmb(is_rhand=True, emb_dim=20).emb.detach().cpu()
        # obj = ArticulatedObject(obj_mesh_path)
        bps_obj_orig = self.encode_bps(obj_mesh_path, return_decoded=True)

        r_vert_map, r_c_map = [], []
        l_vert_map, l_c_map = [], []

        for fr in range(num_frames):
            # Update and load object
            curr_obj_seq = obj_seq[fr]
            curr_obj_articulation, curr_obj_rot, curr_obj_trans = curr_obj_seq[0], curr_obj_seq[1:4], curr_obj_seq[4:]

            # Update BPS points
            curr_bps_pts = self._transform_bps(bps_obj_orig, 
                                          curr_obj_articulation=curr_obj_articulation, 
                                          curr_obj_rot=curr_obj_rot, 
                                          curr_obj_trans=curr_obj_trans
                                          )

            # Update MANO model and load mesh
            left_mano.update(
                pose=left_hand_seq['pose'][fr],
                global_orient=left_hand_seq['rot'][fr],
                trans=left_hand_seq['trans'][fr],
            )
            right_mano.update(
                pose=right_hand_seq['pose'][fr],
                global_orient=right_hand_seq['rot'][fr],
                trans=right_hand_seq['trans'][fr],
            )

            left_mano_mesh, right_mano_mesh = left_mano.get_mesh(), right_mano.get_mesh()

            obj_pts = torch.cat([curr_bps_pts["top_decoded"], curr_bps_pts["bottom_decoded"]], dim=-2)
            obj_pts = obj_pts.cpu().numpy()

            lhand_verts = np.asarray(left_mano_mesh.vertices)
            rhand_verts = np.asarray(right_mano_mesh.vertices)
            tmp_r_v_map, _, tmp_r_c_map = emb_map_gen(obj_pts, rhand_verts, right_emb)
            tmp_l_v_map, _, tmp_l_c_map = emb_map_gen(obj_pts, lhand_verts, left_emb)

            r_vert_map.append(tmp_r_v_map)
            r_c_map.append(tmp_r_c_map)

            l_vert_map.append(tmp_l_v_map)
            l_c_map.append(tmp_l_c_map)

        r_vert_map = np.array(r_vert_map)
        r_c_map = np.array(r_c_map)

        l_vert_map = np.array(l_vert_map)
        l_c_map = np.array(l_c_map)

        print()

        return {
            "r_vert_map": r_vert_map,
            "r_c_map": r_c_map,
            "l_vert_map": l_vert_map,
            "l_c_map": l_c_map,
        }
    
    def _setup_gt_ce_map(self):
        """
        Save the ground truth contact and embedding maps.
        """
        os.makedirs("./data/ce_maps/", exist_ok=True)
        print("Computing and saving ground truth contact and embedding maps...")
        self.sequences["gt_ce_maps"] = []

        for i in tqdm(range(len(self.sequences["mano_seq"]))):
            mano_seq_path = self.sequences["mano_seq"][i]
            obj_seq_path = self.sequences["object_seq"][i]
            obj_mesh_path = self.sequences["object_mesh_path"][i]

            subject_name = mano_seq_path.split("/")[-2]
            fname = mano_seq_path.split("/")[-1].split(".")[0]
            ce_map_fn = os.path.join("./data/ce_maps/", f"{subject_name}_{fname}.npy")
            self.sequences["gt_ce_maps"].append(ce_map_fn)

            if os.path.exists(ce_map_fn):
                continue

            mano_seq = np.load(mano_seq_path, allow_pickle=True).item()
            obj_seq = np.array(np.load(obj_seq_path, allow_pickle=True))
            ce_maps = self._compute_gt_ce_map_from_one_seq(obj_mesh_path, obj_seq, mano_seq)
            np.save(ce_map_fn, ce_maps)

    def _preprocess_sequences(self):
        """
        Preprocess the sequences to compute velocities and canonicalize the object and hand sequences.
        """
        os.makedirs("./data/canon_seqs/", exist_ok=True)
        print("Preprocessing sequences (computing object velocities, canonicalizing the hand positions and rotations)...")
        self.sequences["canon_seq"] = []

        for i in tqdm(range(len(self.sequences["mano_seq"]))):
            mano_seq_path = self.sequences["mano_seq"][i]
            obj_seq_path = self.sequences["object_seq"][i]
            subject_name = mano_seq_path.split("/")[-2]
            fname = mano_seq_path.split("/")[-1].split(".")[0]
            canon_seq_fn = os.path.join("./data/canon_seqs/", f"{subject_name}_{fname}.npy")
            self.sequences["canon_seq"].append(canon_seq_fn)

            if os.path.exists(canon_seq_fn):
                continue

            mano_seq = np.load(mano_seq_path, allow_pickle=True).item()
            obj_seq = np.array(np.load(obj_seq_path, allow_pickle=True))
            canonicalized_seq = self.canonicalize_sequences(mano_seq, obj_seq)
            np.save(canon_seq_fn, canonicalized_seq)


    def _sample_frames_random(self, total_frames, num_sample_frames=120):
        """
        Sample `num_sample_frames` frames randomly from the total frames, while clipping near start and 
        end to avoid 'silent' sequences as the paper.
        """
        if num_sample_frames == -1:
            return 0, total_frames
        clip_bdary = int(0.1 * total_frames)
        start_idx = np.random.randint(clip_bdary, total_frames - num_sample_frames - clip_bdary)
        end_idx = start_idx + num_sample_frames
        return start_idx, end_idx


    def __getitem__(self, index):
        item_dict = {}
        index = index % len(self.sequences["mano_seq"])
        obj_seq_path = self.sequences["object_seq"][index]
        obj_seq = np.array(np.load(obj_seq_path, allow_pickle=True))
        obj_mesh_path = self.sequences["object_mesh_path"][index]
        indices = self._sample_frames_random(obj_seq.shape[0], num_sample_frames=self._return_fixed_length)

        if "mano_seq" in self._return_items:
            mano_seq_path = self.sequences["mano_seq"][index]
            mano_seq = np.load(mano_seq_path, allow_pickle=True).item()
            item_dict["mano_seq"] = mano_seq
        
        if "obj_seq" in self._return_items:
            item_dict["obj_seq"] = obj_seq
        
        if "obj_mesh_path" in self._return_items:
            item_dict["obj_mesh_path"] = obj_mesh_path

        if "bps" in self._return_items:
            bps = self.encode_bps(obj_mesh_path, return_decoded=False)
            item_dict["bps"] = torch.cat([bps["top"], bps["bottom"]], dim=-2).squeeze(0)
        
        if "bps_vis" in self._return_items:
            bps = self.encode_bps(obj_mesh_path, return_decoded=True)
            if "bps" not in item_dict:
                item_dict["bps"] = bps
            else:
                item_dict["bps_vis"] = bps

        if "ce_seq" in self._return_items:
            gt_ce_map_path = self.sequences["gt_ce_maps"][index]
            ce_seq = np.load(gt_ce_map_path, allow_pickle=True).item()

            min_rc, max_rc = np.min(ce_seq["r_c_map"]), np.max(ce_seq["r_c_map"])
            min_lc, max_lc = np.min(ce_seq["l_c_map"]), np.max(ce_seq["l_c_map"])
            if min_rc < 0 or max_rc > 1 or min_lc < 0 or max_lc > 1:
                print(f"ERROR:: {obj_seq_path=}, {min_rc=}, {max_rc=}, {min_lc=}, {max_lc=}")

            r_c_map = torch.tensor(ce_seq["r_c_map"][indices[0]:indices[1]]).float()
            l_c_map = torch.tensor(ce_seq["l_c_map"][indices[0]:indices[1]]).float()
            r_vert_map = ce_seq["r_vert_map"][indices[0]:indices[1]]
            l_vert_map = ce_seq["l_vert_map"][indices[0]:indices[1]]

            min_rc, max_rc = np.min(r_c_map.numpy()), np.max(r_c_map.numpy())
            min_lc, max_lc = np.min(l_c_map.numpy()), np.max(l_c_map.numpy())
            if min_rc < 0 or max_rc > 1 or min_lc < 0 or max_lc > 1:
                print(f"ERROR:: {obj_seq_path=}, {min_rc=}, {max_rc=}, {min_lc=}, {max_lc=}")

            r_e_map = (self.right_emb[r_vert_map])
            l_e_map = self.left_emb[l_vert_map]
            r_e_map[r_c_map < 0.5] = torch.zeros(self.cse_dim)
            l_e_map[l_c_map < 0.5] = torch.zeros(self.cse_dim)

            r_c_map = r_c_map.unsqueeze(-1).unsqueeze(-1)
            r_e_map = r_e_map.unsqueeze(-2)
            r_ce_map = torch.cat([r_c_map, r_e_map], dim=-1)

            l_e_map = l_e_map.unsqueeze(-2)
            l_c_map = l_c_map.unsqueeze(-1).unsqueeze(-1)
            l_ce_map = torch.cat([l_c_map, l_e_map], dim=-1)

            ce_map = torch.cat([r_ce_map, l_ce_map], dim=-2).squeeze(1) # [T, N, 2, 1 + cse_dim]
            item_dict["ce_seq"] = ce_map

        if "canon_seq" in self._return_items:
            mano_seq_path = self.sequences["mano_seq"][index]
            mano_seq = np.load(mano_seq_path, allow_pickle=True).item()
            item_dict["betas"] = torch.cat([
                torch.tensor(mano_seq["right"]["shape"]).float().unsqueeze(0), 
                torch.tensor(mano_seq["left"]["shape"]).float().unsqueeze(0)
                ], dim=-2).float()

            canon_seq = np.load(self.sequences["canon_seq"][index], allow_pickle=True).item()

            rh_pose = torch.tensor(canon_seq["right"][indices[0]:indices[1]]).float().unsqueeze(-2)
            lh_pose = torch.tensor(canon_seq["left"][indices[0]:indices[1]]).float().unsqueeze(-2)
            hand_pose = torch.cat([rh_pose, lh_pose], dim=-2)

            obj_trans_vel = torch.tensor(canon_seq["obj_linear_velocities"][indices[0]:indices[1]]).float()
            obj_angular_vel = torch.tensor(canon_seq["obj_angular_velocities"][indices[0]:indices[1]]).float()
            obj_artic_vel = torch.tensor(canon_seq["obj_artic_velocities"][indices[0]:indices[1]]).float()
            obj_start_pose = obj_seq[indices[0]]
            start_obj_artic, start_obj_rot, start_obj_trans = obj_start_pose[0], obj_start_pose[1:4], obj_start_pose[4:]

            start_obj_trans = torch.tensor(start_obj_trans).float() * 0.001 # convert to m from mm
            start_obj_rot = R.from_rotvec(start_obj_rot).as_matrix()
            start_obj_rot = matrix_to_rotation_6d(torch.tensor(start_obj_rot)).float()

            start_obj_artic = R.from_rotvec(np.array([0, 0, -start_obj_artic])).as_matrix() # need to reverse angle
            start_obj_artic = matrix_to_rotation_6d(torch.tensor(start_obj_artic)).float()

            obj_trans_vel = torch.cat([start_obj_trans.unsqueeze(0), obj_trans_vel], dim=-2)
            obj_angular_vel = torch.cat([start_obj_rot.unsqueeze(0), obj_angular_vel], dim=-2)
            obj_artic_vel = torch.cat([start_obj_artic.unsqueeze(0), obj_artic_vel], dim=-2)
            
            item_dict["hand_pose"] = hand_pose
            item_dict["trans_v"] = obj_trans_vel
            item_dict["angular_v"] = obj_angular_vel
            item_dict["artic_v"] = obj_artic_vel

        item_dict["indices"] = indices
        return item_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic_path", type=str,
                        default="/home/zareef/Datasets/arctic/unpack/arctic_data/",
                        # required=True, 
                        help="Path to unpacked arctic dataset.")
    args = parser.parse_args()
    
    ds = ArcticDataloader(args.arctic_path, split="train", return_fixed_length=120, return_items=["bps", "ce_seq", "canon_seq"])
    # ds = ArcticDataloader(args.arctic_path, split="val", return_fixed_length=120, return_items=["bps", "ce_seq", "canon_seq"])
    # ds = ArcticDataloader(args.arctic_path, split="test", return_fixed_length=120, return_items=["bps", "ce_seq", "canon_seq"])

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    for i, dl_at_i in enumerate(dl):
        # print(f"{i}. {dl_at_i['ce_seq'].shape=}, {dl_at_i['bps'].shape=}, {dl_at_i['hand_pose'].shape=}, \
        #       {dl_at_i['artic_v'].shape}, {dl_at_i['trans_v'].shape=}, {dl_at_i['angular_v'].shape=}")
        print(f"{dl_at_i['hand_pose'].shape=}, {dl_at_i['betas'].shape=} {dl_at_i['bps'].shape=}, {dl_at_i['artic_v'].shape=}, {dl_at_i['trans_v'].shape=}, {dl_at_i['angular_v'].shape=}")