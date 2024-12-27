# Visualizing Result from Baseline MDM training
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
import open3d as o3d
import pytorch3d
from pytorch3d.io import load_obj, load_ply
from bps_torch.bps import bps_torch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from scipy.spatial.transform import Rotation as R

from data.arctic_dataloader import ArcticDataloader
from utils.o3d_utils import *
from utils.utils import *
from utils.mano_helper import MANOWrapper
from training.mano_cse import MANOSurfaceEmb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from visualization.ce_map_vis import project_embeddings_to_colors, rgb_color_from_intensity

from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d

from data.canon_seq import canon_seq, uncanon_seq, uncanon_seq_batch, uncanon_obj_pose_batch

import torch.nn.functional as F


def o3d_pcd(points, color=[0.4, 0.4, 0.4], sphere_pcd=False, radius=1):
    """
    Convert a numpy array of points to an Open3D point cloud.
    """
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    points = points.reshape(-1, 3)

    color = np.array(color)
    if color.ndim == 3:
        color = color.reshape(-1, 3)
    

    if sphere_pcd:
        mesh = o3d.geometry.TriangleMesh()
        for i, pt in enumerate(points):
            if color.ndim == 2:
                tmp_color = color[i]
            else:
                tmp_color = color
            mesh += get_sphere(pt, radius=radius, color=tmp_color)
        mesh.compute_vertex_normals()
        return mesh
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        return pcd


def motion_vis(bps_dict, obj_seq, gt_mano_seq, pred_mano_seq, betas):
    top_decoded = bps_dict["top_decoded"]
    bottom_decoded = bps_dict["bottom_decoded"]

    obj_seq = obj_seq.numpy()[0]
    rh_betas, lh_betas = betas[:, 0], betas[:, 1]

    gt_mano_seq_cpu = {
        "left": {
            "pose": gt_mano_seq["left"]["pose"].detach().cpu().numpy()[0],
            "rot": gt_mano_seq["left"]["rot"].detach().cpu().numpy()[0],
            "trans": gt_mano_seq["left"]["trans"].detach().cpu().numpy()[0],
        },
        "right": {
            "pose": gt_mano_seq["right"]["pose"].detach().cpu().numpy()[0],
            "rot": gt_mano_seq["right"]["rot"].detach().cpu().numpy()[0],
            "trans": gt_mano_seq["right"]["trans"].detach().cpu().numpy()[0],
        }
    }
    pred_mano_seq_cpu = {
        "left": {
            "pose": pred_mano_seq["left"]["pose"].detach().cpu().numpy()[0],
            "rot": pred_mano_seq["left"]["rot"].detach().cpu().numpy()[0],
            "trans": pred_mano_seq["left"]["trans"].detach().cpu().numpy()[0],
        },
        "right": {
            "pose": pred_mano_seq["right"]["pose"].detach().cpu().numpy()[0],
            "rot": pred_mano_seq["right"]["rot"].detach().cpu().numpy()[0],
            "trans": pred_mano_seq["right"]["trans"].detach().cpu().numpy()[0],
        }
    }

    lh_gt_seq, rh_gt_seq = gt_mano_seq_cpu["left"], gt_mano_seq_cpu["right"]
    lh_gt_mano = MANOWrapper(
        is_rhand=False,
        flat_hand_mean=False,
        betas=lh_betas.numpy(),
        init_pose=lh_gt_seq['pose'][0],
        init_orient=lh_gt_seq['rot'][0],
        init_trans=lh_gt_seq['trans'][0],
    )
    rh_gt_mano = MANOWrapper(
        is_rhand=True,
        flat_hand_mean=False,
        betas=rh_betas.numpy(),
        init_pose=rh_gt_seq['pose'][0],
        init_orient=rh_gt_seq['rot'][0],
        init_trans=rh_gt_seq['trans'][0],
    )

    lh_pred_seq, rh_pred_seq = pred_mano_seq_cpu["left"], pred_mano_seq_cpu["right"]
    lh_pred_mano = MANOWrapper(
        is_rhand=False,
        flat_hand_mean=False,
        betas=lh_betas.numpy(),
        init_pose=lh_pred_seq['pose'][0],
        init_orient=lh_pred_seq['rot'][0],
        init_trans=lh_pred_seq['trans'][0],
    )
    rh_pred_mano = MANOWrapper(
        is_rhand=True,
        flat_hand_mean=False,
        betas=rh_betas.numpy(),
        init_pose=rh_pred_seq['pose'][0],
        init_orient=rh_pred_seq['rot'][0],
        init_trans=rh_pred_seq['trans'][0],
    )

    num_frames = obj_seq.shape[0]
    frame_seq = adjust_playback_speed(0, num_frames-1, 1)
    viewer = simpleViewer("Simple Open3D Viewer", 3840, 2160)
    viewer.total_frames = len(frame_seq)
    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    viewer.add_geometry({"name":"global", "geometry":global_coord})
    viewer.add_plane()

    for fr in tqdm(frame_seq):
        curr_obj_seq = obj_seq[fr]
        curr_obj_articulation, curr_obj_rot, curr_obj_trans = curr_obj_seq[0], curr_obj_seq[1:4], curr_obj_seq[4:]
        upd_bps_dict = ds._transform_bps(bps_dict, curr_obj_rot, curr_obj_trans, curr_obj_articulation)

        top_decoded, bottom_decoded = upd_bps_dict["top_decoded"], upd_bps_dict["bottom_decoded"]
        top_decoded, bottom_decoded = top_decoded.cpu().numpy(), bottom_decoded.cpu().numpy()
        decoded_pcd = np.concatenate([top_decoded, bottom_decoded], axis=0)

        decoded_pcd_o3d_gt = o3d_pcd(decoded_pcd, sphere_pcd=True, radius=0.002)
        decoded_pcd_o3d_pred = deepcopy(decoded_pcd_o3d_gt)
        # viewer.remove_geometry(geom_name="object_pred")
        # viewer.add_geometry({"name":"object_pred", "geometry":decoded_pcd_o3d_pred})
        lh_gt_mano.update(
            pose=lh_gt_seq['pose'][fr],
            global_orient=lh_gt_seq['rot'][fr],
            trans=lh_gt_seq['trans'][fr],
        )
        rh_gt_mano.update(
            pose=rh_gt_seq['pose'][fr],
            global_orient=rh_gt_seq['rot'][fr],
            trans=rh_gt_seq['trans'][fr],
        )
        lh_gt_mesh, rh_gt_mesh = lh_gt_mano.get_mesh(), rh_gt_mano.get_mesh()
        lh_gt_mesh.translate([0.5, 0., 0.])
        rh_gt_mesh.translate([0.5, 0., 0.])
        decoded_pcd_o3d_gt.translate([0.5, 0., 0.])

        viewer.remove_geometry(geom_name="lh_gt")
        viewer.remove_geometry(geom_name="rh_gt")
        viewer.add_geometry({"name":"lh_gt", "geometry":lh_gt_mesh})
        viewer.add_geometry({"name":"rh_gt", "geometry":rh_gt_mesh})
        viewer.remove_geometry(geom_name="object_gt")
        viewer.add_geometry({"name":"object_gt", "geometry":decoded_pcd_o3d_gt})


        lh_pred_mano.update(
            pose=lh_pred_seq['pose'][fr],
            global_orient=lh_pred_seq['rot'][fr],
            trans=lh_pred_seq['trans'][fr],
        )
        rh_pred_mano.update(
            pose=rh_pred_seq['pose'][fr],
            global_orient=rh_pred_seq['rot'][fr],
            trans=rh_pred_seq['trans'][fr],
        )
        lh_pred_mesh, rh_pred_mesh = lh_pred_mano.get_mesh(), rh_pred_mano.get_mesh()

        viewer.remove_geometry(geom_name="lh_pred")
        viewer.remove_geometry(geom_name="rh_pred")
        viewer.add_geometry({"name":"lh_pred", "geometry":lh_pred_mesh})
        viewer.add_geometry({"name":"rh_pred", "geometry":rh_pred_mesh})
        viewer.remove_geometry(geom_name="object_pred")
        viewer.add_geometry({"name":"object_pred", "geometry":decoded_pcd_o3d_pred})
        viewer.tick()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic_path", type=str,
                        default="/home/zareef/Datasets/arctic/unpack/arctic_data/",
                        help="Path to unpacked arctic dataset.")
    parser.add_argument("--sample_path", default="models/samples/baseline/sample_scissor.pkl",
                         type=str, help="Path to the generated sample")
    args = parser.parse_args()
    ds = ArcticDataloader(data_root=args.arctic_path, split="val")

    import pickle 
    with open(args.sample_path, "rb") as f:
        sample = pickle.load(f)

    print(f"{sample.keys()=}, {sample['obj_motion']['bps'].shape=}")

    motion_vis(
        bps_dict=sample["bps_vis"],
        obj_seq=sample["pred_obj"],
        gt_mano_seq=sample["gt_mano"],
        pred_mano_seq=sample["pred_mano"],
        betas=sample["betas"],
    )