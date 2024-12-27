# Visualizing Result from Stage 1 CE Map Generation training
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

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


def vis_bps_ce_map(bps_dict, trans_v, angular_v, artic_v, gt_ce_map, pred_ce_map, hand="right", vis="dist"):
    top_decoded = bps_dict["top_decoded"]
    bottom_decoded = bps_dict["bottom_decoded"]

    obj_seq = uncanon_obj_pose_batch(trans_v, angular_v, artic_v)
    obj_seq = obj_seq.numpy()[0]
    print(obj_seq.shape)

    gt_ce_map = gt_ce_map[0]
    pred_ce_map = pred_ce_map[0]
    
    r_c_map_gt, l_c_map_gt  = gt_ce_map[:, :, 0, 0], gt_ce_map[:, :, 1, 0]
    r_c_map_pred, l_c_map_pred = pred_ce_map[:, :, 0, 0], pred_ce_map[:, :, 1, 0]
    r_c_map_pred = F.sigmoid(r_c_map_pred)
    l_c_map_pred = F.sigmoid(l_c_map_pred)
    # r_c_map_gt[r_c_map_gt > 0.5] = 1
    # l_c_map_gt[l_c_map_gt > 0.5] = 1
    r_c_map_pred[r_c_map_pred > 0.4] = 1
    l_c_map_pred[l_c_map_pred > 0.4] = 1



    num_frames = obj_seq.shape[0]
    frame_seq = adjust_playback_speed(0, num_frames-1, 1)
    viewer = simpleViewer("Simple Open3D Viewer", 3840, 2160)
    viewer.total_frames = len(frame_seq)
    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    viewer.add_geometry({"name":"global", "geometry":global_coord})
    viewer.add_plane()

    for fr in tqdm(frame_seq):
        r_c_map_gt_fr, l_c_map_gt_fr = r_c_map_gt[fr], l_c_map_gt[fr]
        r_c_map_pred_fr, l_c_map_pred_fr = r_c_map_pred[fr], l_c_map_pred[fr]
        curr_obj_seq = obj_seq[fr]
        curr_obj_articulation, curr_obj_rot, curr_obj_trans = curr_obj_seq[0], curr_obj_seq[1:4], curr_obj_seq[4:]
        upd_bps_dict = ds._transform_bps(bps_dict, curr_obj_rot, curr_obj_trans, curr_obj_articulation)

        top_decoded, bottom_decoded = upd_bps_dict["top_decoded"], upd_bps_dict["bottom_decoded"]
        top_decoded, bottom_decoded = top_decoded.cpu().numpy(), bottom_decoded.cpu().numpy()
        decoded_pcd = np.concatenate([top_decoded, bottom_decoded], axis=0)
        if hand == "right":
            if vis == "emb":
                # obj_colors = right_colors[r_vert_map_fr]
                obj_colors = [0.4, 0.4, 0.4]
            else:
                obj_colors_gt = rgb_color_from_intensity(r_c_map_gt_fr)
                obj_colors_pred = rgb_color_from_intensity(r_c_map_pred_fr)
        else:
            if vis == "emb":
                # obj_colors = left_colors[l_vert_map_fr]
                obj_colors = [0.4, 0.4, 0.4]
            else:
                obj_colors_gt = rgb_color_from_intensity(l_c_map_gt_fr)
                obj_colors_pred = rgb_color_from_intensity(l_c_map_pred_fr)

        decoded_pcd_o3d_pred = o3d_pcd(decoded_pcd, color=obj_colors_pred, sphere_pcd=True, radius=0.002)
        decoded_pcd_o3d_gt = o3d_pcd(decoded_pcd, color=obj_colors_gt, sphere_pcd=True, radius=0.002)
        decoded_pcd_o3d_gt.translate([0.5, 0., 0.])
        viewer.remove_geometry(geom_name="object_pred")
        viewer.add_geometry({"name":"object_pred", "geometry":decoded_pcd_o3d_pred})
        viewer.remove_geometry(geom_name="object_gt")
        viewer.add_geometry({"name":"object_gt", "geometry":decoded_pcd_o3d_gt})
        viewer.tick()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic_path", type=str,
                        default="/home/zareef/Datasets/arctic/unpack/arctic_data/",
                        help="Path to unpacked arctic dataset.")
    parser.add_argument("--sample_path", default="models/samples/stage1/sample_espresso.pkl",
                         type=str, help="Path to the generated sample")
    args = parser.parse_args()

    import pickle 
    with open(args.sample_path, "rb") as f:
        sample = pickle.load(f)

    print(f"{sample.keys()=}, {sample['obj_motion']['bps'].shape=}")

    obj_motion = sample["obj_motion"]
    trans_v, angular_v, artic_v = obj_motion["trans_v"], obj_motion["angular_v"], obj_motion["artic_v"]
    ds = ArcticDataloader(data_root=args.arctic_path, split="val")

    vis_bps_ce_map(
        bps_dict=sample["bps_vis"],
        trans_v=trans_v,
        angular_v=angular_v,
        artic_v=artic_v,
        gt_ce_map=sample["gt"],
        pred_ce_map=sample["pred"],

    )