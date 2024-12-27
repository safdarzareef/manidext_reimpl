# Visualize Ground Truth Contact and Embedding Maps
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import copy
import argparse

from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from utils.utils import find_nearest_vertices
from utils.mano_helper import MANOWrapper
from utils.artic_obj_helper import ArticulatedObject
from utils.o3d_utils import simpleViewer, adjust_playback_speed
from training.mano_cse import MANOSurfaceEmb

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def project_embeddings_to_colors(embeddings, method='pca'):
    """
    Project high-dimensional embeddings to RGB colors
    
    Args:
        embeddings: numpy array of shape (N, dim) where dim > 3
        method: str, either 'pca' or 'direct'
    
    Returns:
        colors: numpy array of shape (N, 3) with values in [0,1]
    """
    if method == 'pca':
        # Use PCA to reduce to 3 dimensions
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings)
    else:
        # Take first 3 principal components
        reduced = embeddings[:, :3]
    
    # Scale to [0,1] range
    scaler = MinMaxScaler()
    colors = scaler.fit_transform(reduced)
    return colors


import matplotlib.pyplot as plt

def rgb_color_from_intensity(intensity, colormap='plasma'):
    """
    Convert intensity values to RGB colors using a matplotlib colormap.
    
    Parameters:
    -----------
    intensity : float or array-like
        Single value or array of values between 0 and 1
    colormap : str, default='plasma'
        Name of matplotlib colormap to use
        
    Returns:
    --------
    numpy.ndarray
        RGB colors with shape (..., 3) where ... matches input shape
    """
    # Convert input to numpy array
    intensity = np.asarray(intensity)
    
    # Clip values to [0, 1]
    intensity = np.clip(intensity, 0, 1)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Handle single values vs batches
    if intensity.ndim == 0:
        return cmap(float(intensity))[:3]
    
    # Map colors and remove alpha channel
    colors = cmap(intensity.ravel())[:, :3]
    
    # Reshape to match input dimensions
    return colors.reshape(intensity.shape + (3,))


def emb_map_gen(obj_verts, hand_verts, hand_embs, contact_sigma=0.25):
    vert_mapping, vert_dist = find_nearest_vertices(obj_verts, hand_verts)
    vert_embs = hand_embs[vert_mapping]
    vert_dist = np.array(vert_dist)
    
    if np.min(vert_dist) > 0.0075:
        contact_map = np.zeros_like(vert_dist)
    else:
        vert_dist = (vert_dist - np.min(vert_dist)) / (np.max(vert_dist) - np.min(vert_dist))
        
        contact_map = np.exp(-vert_dist/(2 * contact_sigma**2))
        contact_map = np.clip(contact_map, 0, 1)
    return vert_mapping, vert_embs, contact_map


def vis_arctic_ce_map(obj_seq, mano_seq, obj_mesh_path, show_emb=True, show_rhand=False):
    """
    Visualizes the Arctic dataset sequence with articulated object, and MANO hands.\
    
    The object is colored based on a continuous surface embedding (CSE) map.

    Args:
        obj_seq (np.ndarray): ARCTIC object sequence with shape (num_frames, 7).
        mano_seq (np.ndarray): ARCTIC MANO sequence with poses, global position and orientation.
        obj_mesh_path (str): Path where the top and bottom object meshes are stored.
    """
    print("Visualizing Arctic dataset sequence...")
    print(f"Object sequence {obj_seq.shape}")

    num_frames = obj_seq.shape[0]
    frame_seq = adjust_playback_speed(0, num_frames-1, 4)

    print(f"Number of frames {num_frames}")

    mano_seq = mano_seq.item()
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
    left_colors = project_embeddings_to_colors(left_emb, method='pca')

    right_mano = MANOWrapper(
            is_rhand=True,
            flat_hand_mean=False,
            betas=right_hand_seq['shape'],
            init_pose=right_hand_seq['pose'][0],
            init_orient=right_hand_seq['rot'][0],
            init_trans=right_hand_seq['trans'][0],
        )
    right_emb = MANOSurfaceEmb(is_rhand=True, emb_dim=20).emb.detach().cpu()
    right_colors = project_embeddings_to_colors(right_emb, method='pca')

    green_color = np.array([80, 210, 80]) * (1/255)
    pink_color = np.array([255, 80, 80]) * (1/255)
    
    obj = ArticulatedObject(obj_mesh_path)

    viewer = simpleViewer("Simple Open3D Viewer", 3860, 2140)
    viewer.total_frames = len(frame_seq)
    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    viewer.add_geometry({"name":"global", "geometry":global_coord})
    viewer.add_plane()

    for fr in tqdm(frame_seq):
        # if fr < 0.1*num_frames or fr > num_frames - 0.1*num_frames: 
        #     continue
        # Update and load object
        curr_obj_seq = obj_seq[fr]
        curr_obj_articulation, curr_obj_rot, curr_obj_trans = curr_obj_seq[0], curr_obj_seq[1:4], curr_obj_seq[4:]

        obj.transform_overall(curr_obj_rot, curr_obj_trans, curr_obj_articulation)
        obj_mesh = obj.get_mesh()

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
        
        obj_verts = np.asarray(obj_mesh.vertices)
        lhand_verts = np.asarray(left_mano_mesh.vertices)
        rhand_verts = np.asarray(right_mano_mesh.vertices)

        if show_rhand:
            r_hand_mapping, r_e_map, r_c_map = emb_map_gen(obj_verts, rhand_verts, right_emb)
            left_mano_mesh.paint_uniform_color(green_color)
            if show_emb:
                right_mano_mesh.vertex_colors = o3d.utility.Vector3dVector(right_colors)
                obj_colors = right_colors[r_hand_mapping]
                obj_colors[r_c_map < 0.5] = [0.4, 0.4, 0.4]
                obj_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_colors)
            else:
                obj_colors = rgb_color_from_intensity(r_c_map)
                obj_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_colors)
                right_mano_mesh.paint_uniform_color(pink_color)
        else:
            l_hand_mapping, l_e_map, l_c_map = emb_map_gen(obj_verts, lhand_verts, left_emb)
            right_mano_mesh.paint_uniform_color(green_color)
            if show_emb:
                left_mano_mesh.vertex_colors = o3d.utility.Vector3dVector(left_colors)
                obj_colors = left_colors[l_hand_mapping]
                obj_colors[l_c_map < 0.5] = [0.4, 0.4, 0.4]
                obj_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_colors)
            else:
                obj_colors = rgb_color_from_intensity(l_c_map)
                obj_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_colors)
                left_mano_mesh.paint_uniform_color(pink_color)

        viewer.remove_geometry(geom_name="object")
        viewer.add_geometry({"name":"object", "geometry":obj_mesh})
        viewer.remove_geometry(geom_name="left_mano")
        viewer.add_geometry({"name":"left_mano", "geometry":left_mano_mesh})
        viewer.remove_geometry(geom_name="right_mano")
        viewer.add_geometry({"name":"right_mano", "geometry":right_mano_mesh})

        viewer.tick()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic_path", type=str,
                        default="/home/zareef/Datasets/arctic/unpack/arctic_data/",
                        # required=True, 
                        help="Path to unpacked arctic dataset.")
    parser.add_argument("--subject", type=str, default="s01", help="Subject id e.g. s01, s02, ... to visualize.")
    parser.add_argument("--object", type=str, default="box", help="Object name e.g. box, capsulemachine, ketchup, ... to visualize.")
    parser.add_argument("--action", type=str, default="use", help="Action type (grab or use) to visualize.")
    parser.add_argument("--seq_num", type=str, default="01", help="Sequence number to visualize (e.g. 01, 02, 03, 04).")
    parser.add_argument("--rhand", action="store_true", help="Use right hand model else left hand")
    parser.add_argument("--vis_emb", action="store_true", help="Visualize embeddings")

    args = parser.parse_args()

    arctic_filename_prefix = f"{args.object}_{args.action}_{args.seq_num}"
    obj_seq = np.load(os.path.join(args.arctic_path, f"data/raw_seqs/{args.subject}/{arctic_filename_prefix}.object.npy"),
                      allow_pickle=True)
    mano_seq = np.load(os.path.join(args.arctic_path, f"data/raw_seqs/{args.subject}/{arctic_filename_prefix}.mano.npy"),
                      allow_pickle=True)
    smplx_seq = np.load(os.path.join(args.arctic_path, f"data/raw_seqs/{args.subject}/{arctic_filename_prefix}.smplx.npy"),
                      allow_pickle=True)

    obj_mesh_path = os.path.join(args.arctic_path, f"data/meta/object_vtemplates/{args.object}/")

    vis_arctic_ce_map(obj_seq, mano_seq, obj_mesh_path, show_emb=args.vis_emb, show_rhand=args.rhand)
