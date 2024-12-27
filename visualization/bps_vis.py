# Visualizing BPS Representation from the dataloader and corresponding maps and sequences to make sure it works.
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

from data.canon_seq import canon_seq, uncanon_seq, uncanon_seq_batch

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


def vis_bps(mesh_path,n_pts = 4096, vis=False):
    print(f"Loading mesh from {mesh_path}")
    mesh1 = load_mesh_pytorch3d(os.path.join(mesh_path, "top.obj"))
    mesh2 = load_mesh_pytorch3d(os.path.join(mesh_path, "bottom.obj"))
    mesh = join_meshes_as_batch([mesh1, mesh2]).to("cuda")
    
    bps = bps_torch(
        bps_type="grid_sphere",
        n_bps_points=n_pts,
        radius=0.3,
    )

    sampled_pcd = pytorch3d.ops.sample_points_from_meshes(mesh, n_pts) * (0.001)
    print(f"{sampled_pcd.shape=}, {sampled_pcd.min()=}, {sampled_pcd.max()=}")
    bps_pcd = bps.encode(sampled_pcd,
                     feature_type=['dists','deltas'],
                     custom_basis=None
                     )['deltas']
    decoded_pcd = bps.decode(bps_pcd)

    if vis:
        sampled_pc_o3d = o3d_pcd(sampled_pcd, color=[0.4, 0.4, 0.4])
        decoded_pcd_o3d = o3d_pcd(decoded_pcd, color=[1, 0., 0.])

        o3d.visualization.draw_geometries(
            [decoded_pcd_o3d],
            window_name="Mesh Visualization",
            width=800,
            height=600,
            point_show_normal=False
        )
    return bps_pcd, decoded_pcd


def get_vert_map_from_emb_map(emb_map, emb_to_vert):
    """
    Get the vertex map from the embedding map and the embedding to vertex map.
    """
    emb_map = emb_map.cpu().numpy()
    emb_map = emb_map.reshape(-1, emb_map.shape[-1])
    emb_to_vert = emb_to_vert.cpu().numpy()
    emb_to_vert = emb_to_vert.reshape(-1, emb_to_vert.shape[-1])
    vert_map = np.zeros((emb_map.shape[0], 3))
    for i in range(emb_map.shape[0]):
        vert_map[i] = emb_to_vert[np.argmin(np.linalg.norm(emb_to_vert - emb_map[i], axis=1))]
    return vert_map


def vis_bps_w_seq(ds, idx, hand="right", vis="emb"):
    """
    Visualize the motion and the gt contact and embedding maps using the BPS representation of the object mesh.
    """
    ds_at_idx = ds[idx]
    obj_seq, mano_seq, obj_mesh_path, bps_dict = ds_at_idx["obj_seq"], ds_at_idx["mano_seq"], ds_at_idx["obj_mesh_path"], ds_at_idx["bps"]
    bps_dict = ds_at_idx["bps"]
    hand_pose, trans_v, angular_v, artic_v = ds_at_idx["hand_pose"], ds_at_idx["trans_v"], ds_at_idx["angular_v"], ds_at_idx["artic_v"]
    betas = ds_at_idx["betas"]
    print(f"{betas.shape=}")
    rh_betas, lh_betas = betas[0], betas[1]
    indices = ds_at_idx["indices"]

    obj_seq = obj_seq[indices[0]:indices[1]]
    # new_obj_seq, new_mano_seq = uncanon_seq(hand_pose, trans_v, angular_v, artic_v, orig_obj_seq=obj_seq)

    new_obj_seq, new_mano_seq = uncanon_seq_batch(
        hand_pose.unsqueeze(0),
        trans_v.unsqueeze(0),
        angular_v.unsqueeze(0),
        artic_v.unsqueeze(0),
    )

    mano_seq = new_mano_seq
    obj_seq = new_obj_seq.squeeze(0)

    for k,v in mano_seq["left"].items():
        mano_seq["left"][k] = v.squeeze(0)
    for k,v in mano_seq["right"].items():
        mano_seq["right"][k] = v.squeeze(0)

    ce_seq = ds_at_idx["ce_seq"]
    r_c_map, l_c_map  = ce_seq[:, :, 0, 0], ce_seq[:, :, 1, 0]

    print(f"Visualizing sequence with indices {indices}")
    top_bps = bps_dict["top"]
    bottom_bps = bps_dict["bottom"]
    top_decoded = bps_dict["top_decoded"]
    bottom_decoded = bps_dict["bottom_decoded"]
    num_frames = obj_seq.shape[0]

    decoded_pcd = np.concatenate([top_decoded.cpu().numpy(), bottom_decoded.cpu().numpy()], axis=0)
    decoded_pcd_o3d = o3d_pcd(decoded_pcd, color=[1, 0., 0.])

    o3d.visualization.draw_geometries(
            [decoded_pcd_o3d],
            window_name="Mesh Visualization",
            width=800,
            height=600,
            point_show_normal=False
        )
    
    left_hand_seq = mano_seq["left"]
    right_hand_seq = mano_seq["right"]
    
    left_mano = MANOWrapper(
            is_rhand=False,
            flat_hand_mean=False,
            betas=lh_betas.numpy(),
            init_pose=left_hand_seq['pose'][0],
            init_orient=left_hand_seq['rot'][0],
            init_trans=left_hand_seq['trans'][0],
        )
    # left_emb = MANOSurfaceEmb(is_rhand=False, emb_dim=20).emb.detach().cpu()
    left_emb = ds.left_emb
    left_colors = project_embeddings_to_colors(left_emb, method='pca')

    right_mano = MANOWrapper(
            is_rhand=True,
            flat_hand_mean=False,
            betas=rh_betas.numpy(),
            init_pose=right_hand_seq['pose'][0],
            init_orient=right_hand_seq['rot'][0],
            init_trans=right_hand_seq['trans'][0],
        )
    # right_emb = MANOSurfaceEmb(is_rhand=True, emb_dim=20).emb.detach().cpu()
    right_emb = ds.right_emb
    right_colors = project_embeddings_to_colors(right_emb, method='pca')
    
    # r_vert_map = get_vert_map_from_emb_map(ce_seq[:, :, 0, 1:], right_emb)
    # l_vert_map = get_vert_map_from_emb_map(ce_seq[:, :, 1, 1:], left_emb)

    frame_seq = adjust_playback_speed(0, num_frames-1, 1)
    viewer = simpleViewer("Simple Open3D Viewer", 3840, 2160)
    viewer.total_frames = len(frame_seq)
    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    viewer.add_geometry({"name":"global", "geometry":global_coord})
    viewer.add_plane()

    for fr in tqdm(frame_seq):
        # Update and load object
        # r_vert_map_fr, r_c_map_fr = r_vert_map[fr], r_c_map[fr]
        # l_vert_map_fr, l_c_map_fr = l_vert_map[fr], l_c_map[fr]
        r_c_map_fr, l_c_map_fr = r_c_map[fr], l_c_map[fr]

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
                obj_colors = rgb_color_from_intensity(r_c_map_fr)
        else:
            if vis == "emb":
                # obj_colors = left_colors[l_vert_map_fr]
                obj_colors = [0.4, 0.4, 0.4]
            else:
                obj_colors = rgb_color_from_intensity(l_c_map_fr)

        decoded_pcd_o3d = o3d_pcd(decoded_pcd, color=obj_colors, sphere_pcd=True, radius=0.002)
        viewer.remove_geometry(geom_name="object")
        viewer.add_geometry({"name":"object", "geometry":decoded_pcd_o3d})

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
        # if hand == "right":
        #     right_mano_mesh.vertex_colors = o3d.utility.Vector3dVector(right_colors)
        # else:
        #     left_mano_mesh.vertex_colors = o3d.utility.Vector3dVector(left_colors)
        viewer.remove_geometry(geom_name="left_mano")
        viewer.add_geometry({"name":"left_mano", "geometry":left_mano_mesh})
        viewer.remove_geometry(geom_name="right_mano")
        viewer.add_geometry({"name":"right_mano", "geometry":right_mano_mesh})

        viewer.tick()
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic_path", type=str,
                        default="/home/zareef/Datasets/arctic/unpack/arctic_data/",
                        help="Path to unpacked arctic dataset.")
    args = parser.parse_args()
    ds = ArcticDataloader(args.arctic_path, split="train", return_fixed_length=240)
    # vis_bps(mesh_path=ds[0]["obj_mesh_path"], vis=True)
    vis_bps_w_seq(ds, 0)