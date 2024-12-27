# Helper MANO class to visualize hand mesh in Open3D.
import numpy as np
import torch
import smplx
import open3d as o3d
from utils.o3d_utils import *
from scipy.spatial.transform import Rotation as R


class MANOWrapper:
    def __init__(self, 
                 model_path="./models", 
                 is_rhand=True,
                 betas=None,
                 init_pose=None,
                 init_orient=None,
                 init_trans=None,
                 flat_hand_mean=True,
                 ):
        self.model = smplx.create(
            model_path=model_path,
            model_type="mano",
            is_rhand=is_rhand,
            use_pca=False, # no pca only use rotation
            flat_hand_mean=flat_hand_mean, # initialize hand pose to flat hand (pose rot zero = flat hand)
        )
        self.is_rhand = is_rhand

        self.betas = torch.zeros(1, 10) if betas is None else torch.from_numpy(betas).float().view(1, 10)
        self.hand_pose = torch.zeros(1, 15, 3)
        self.global_orient = torch.zeros(1, 3)
        self.trans = torch.zeros(1, 3)
        self.update(pose=init_pose, global_orient=init_orient, trans=init_trans)

        self.kinematic_tree = None
        self.joint_dist = torch.zeros(1, 15)
        self.setup_kinematic_tree()
        
    def setup_kinematic_tree(self):
        """
        Joints same order as SMPL-X:
            0   "wrist",
            1   "right_index1",
            2   "right_index2",
            3   "right_index3",
            4   "right_middle1",
            5   "right_middle2",
            6   "right_middle3",
            7   "right_pinky1",
            8   "right_pinky2",
            9   "right_pinky3",
            10  "right_ring1",
            11  "right_ring2",
            12  "right_ring3",
            13  "right_thumb1",
            14  "right_thumb2",
            15  "right_thumb3",
        """
        self.kinematic_tree = [
            None, 
            0, 1, 2, # index
            0, 4 , 5, # middle
            0, 7, 8, # pinky
            0, 10, 11, # ring
            0, 13, 14, # thumb 
            ] # parent joints len(16)
        
        self.joint_dist = np.zeros((1, 15)) # joint distances, indexed by parent joint
        flat_hand_joints = self.get_mano_space_joints_pos()
        for i in range(1, 16):
            parent = self.kinematic_tree[i]
            if parent is not None:
                self.joint_dist[0][i-1] = ((flat_hand_joints[parent] - flat_hand_joints[i]) ** 2).sum() ** 0.5
        self.tmp_joint_pos = flat_hand_joints # for debugging visualization

    def update(self, pose, global_orient, trans):
        if pose is not None:
            if type(pose) == np.ndarray:
                self.hand_pose = torch.from_numpy(pose).float().view(1, 15, 3)
            elif type(pose) == torch.Tensor:
                self.hand_pose = pose.float().view(1, 15, 3)
            else:
                raise ValueError(f"Invalid type ({type(pose)}) for pose, not np.ndarray or torch.Tensor!")
        if global_orient is not None:
            if type(pose) == np.ndarray:
                self.global_orient = torch.from_numpy(global_orient).float().view(1, 3)
            elif type(pose) == torch.Tensor:
                self.global_orient = global_orient.float().view(1, 3)
            else:
                raise ValueError(f"Invalid type ({type(global_orient)}) for global_orient, not np.ndarray or torch.Tensor!")
        if trans is not None:
            if type(pose) == np.ndarray:
                self.trans = torch.from_numpy(trans).float().view(1, 3)
            elif type(pose) == torch.Tensor:
                self.trans = trans.float().view(1, 3)
            else:
                raise ValueError(f"Invalid type ({type(trans)}) for trans, not np.ndarray or torch.Tensor!")

    def get_mano_output(self):
        if self.hand_pose is not None:
            self.hand_pose = self.hand_pose.view(1, -1)
        mano_out = self.model(
                    betas=self.betas, 
                    hand_pose=self.hand_pose, 
                    global_orient=self.global_orient,
                    return_verts=True
                    )
        self.hand_pose = self.hand_pose.view(1, 15, 3)
        return mano_out
    
    def get_mano_space_joints_pos(self):
        mano_out = self.get_mano_output()
        joints = mano_out.joints.detach().cpu().numpy().squeeze()
        return joints
       
    def set_hand_pose_from_angle(self, joint_idx, polar_angle, azimuthal_angle):
        assert(joint_idx < 16)

        if joint_idx == 0:
            self.global_orient[0] = torch.from_numpy(R.from_euler('xyz', [0, azimuthal_angle, polar_angle], degrees=False).as_rotvec())
        else:
            self.hand_pose[0][joint_idx-1] = torch.from_numpy(R.from_euler('xyz', [0, azimuthal_angle, polar_angle], degrees=False).as_rotvec())

    def get_mesh(self):
        mano_out = self.get_mano_output()
        vertices = mano_out.vertices.detach().cpu().numpy().squeeze()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        trans_np = self.trans.flatten().numpy()
        mesh.translate(trans_np)
        return mesh

            
    def _render_tmp_joint_pos(self, highlight_idxs):
        # for debugging visualization only
        mesh = o3d.geometry.TriangleMesh()
        for j in range(self.tmp_joint_pos.shape[0]):
            if j in set(highlight_idxs):
                sphere = get_sphere(self.tmp_joint_pos[j], color=(1,0,0), radius=0.01)
            else:
                sphere = get_sphere(self.tmp_joint_pos[j], color=(0,0,1), radius=0.005)
            mesh += sphere

        for cidx, pidx in enumerate(self.kinematic_tree):
            if pidx == None:
                continue
            parent, child = self.tmp_joint_pos[pidx], self.tmp_joint_pos[cidx]
            mesh += get_segment(parent, child, 0.0025, color=(0.4,0.4,0.4))
        return mesh