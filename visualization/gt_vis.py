# Visualize GT Sequences from Arctic Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import copy
import argparse

from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from utils.mano_helper import MANOWrapper
from utils.artic_obj_helper import ArticulatedObject
from utils.o3d_utils import simpleViewer, adjust_playback_speed


def vis_arctic_seq(obj_seq, mano_seq, obj_mesh_path):
    """
    Visualizes the Arctic dataset sequence with articulated object, and MANO hands.

    Args:
        obj_seq (np.ndarray): ARCTIC object sequence with shape (num_frames, 7).
        mano_seq (np.ndarray): ARCTIC MANO sequence with poses, global position and orientation.
        obj_mesh_path (str): Path where the top and bottom object meshes are stored.
    """
    print("Visualizing Arctic dataset sequence...")
    print(f"Object sequence {obj_seq.shape}")

    num_frames = obj_seq.shape[0]
    frame_seq = adjust_playback_speed(0, num_frames-1, 0.5)

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
    right_mano = MANOWrapper(
            is_rhand=True,
            flat_hand_mean=False,
            betas=right_hand_seq['shape'],
            init_pose=right_hand_seq['pose'][0],
            init_orient=right_hand_seq['rot'][0],
            init_trans=right_hand_seq['trans'][0],
        )
    
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
        viewer.remove_geometry(geom_name="object")
        viewer.add_geometry({"name":"object", "geometry":obj_mesh})

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
    parser.add_argument("--subject", type=str, default="s02", help="Subject id e.g. s01, s02, ... to visualize.")
    parser.add_argument("--object", type=str, default="capsulemachine", help="Object name e.g. box, capsulemachine, ketchup, ... to visualize.")
    parser.add_argument("--action", type=str, default="use", help="Action type (grab or use) to visualize.")
    parser.add_argument("--seq_num", type=str, default="01", help="Sequence number to visualize (e.g. 01, 02, 03, 04).")

    args = parser.parse_args()

    arctic_filename_prefix = f"{args.object}_{args.action}_{args.seq_num}"
    obj_seq = np.load(os.path.join(args.arctic_path, f"data/raw_seqs/{args.subject}/{arctic_filename_prefix}.object.npy"),
                      allow_pickle=True)
    mano_seq = np.load(os.path.join(args.arctic_path, f"data/raw_seqs/{args.subject}/{arctic_filename_prefix}.mano.npy"),
                      allow_pickle=True)
    smplx_seq = np.load(os.path.join(args.arctic_path, f"data/raw_seqs/{args.subject}/{arctic_filename_prefix}.smplx.npy"),
                      allow_pickle=True)

    obj_mesh_path = os.path.join(args.arctic_path, f"data/meta/object_vtemplates/{args.object}/")

    vis_arctic_seq(obj_seq, mano_seq, obj_mesh_path)
