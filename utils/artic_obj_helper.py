import os
import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# Simple articulated object class for visualization and dataloader
class ArticulatedObject: 
    def __init__(self, obj_mesh_path):
        self.obj_mesh_path = obj_mesh_path
        self.obj_mesh_top = o3d.io.read_triangle_mesh(os.path.join(obj_mesh_path, "top.obj"))
        self.obj_mesh_top.paint_uniform_color([0.776, 1.0, 0.7764])
        self.obj_mesh_top.compute_vertex_normals()

        self.obj_mesh_bottom = o3d.io.read_triangle_mesh(os.path.join(obj_mesh_path, "bottom.obj"))
        self.obj_mesh_bottom.paint_uniform_color([1.0, 0.8, 0.8])
        self.obj_mesh_bottom.compute_vertex_normals()

        # Setting current object mesh so original state meshes are preserved
        self.curr_obj_mesh_top = copy.deepcopy(self.obj_mesh_top)
        self.curr_obj_mesh_bottom = copy.deepcopy(self.obj_mesh_bottom)

    def transform_overall(self, global_rot, global_trans, top_artic_rot):
        """
        Transform the object with global rotation and translation, and top part articulation.

        Args:
            global_rot (np.ndarray): Global rotation in axis-angle format.
            global_trans (np.ndarray): Global translation. (in millimeters) 
            top_artic_rot (float): Top part articulation rotation in radians.
        """
        # Reset object to original transform state
        self.curr_obj_mesh_top = copy.deepcopy(self.obj_mesh_top)
        self.curr_obj_mesh_bottom = copy.deepcopy(self.obj_mesh_bottom)

        # For ARCTIC dataset, articulation is rotation around z-axis of the top part
        if top_artic_rot != 0: # just skip if no articulation
            artic_rot_vec = np.array([0, 0, -top_artic_rot]) # need to reverse angle
            artic_rot_mtx = np.eye(4)
            artic_rot_mtx[:3, :3] = R.from_rotvec(artic_rot_vec).as_matrix()
            self.curr_obj_mesh_top.transform(artic_rot_mtx)

        # Global rotation
        global_rot_mtx = np.eye(4)
        # Obj rot in axis-angle format
        global_rot_mtx[:3, :3] = R.from_rotvec(global_rot).as_matrix()

        # Global translation
        global_trans_mtx = np.eye(4)
        global_trans_mtx[:3, 3] = global_trans

        trans_mtx = global_trans_mtx @ global_rot_mtx
        self.curr_obj_mesh_top.transform(trans_mtx)
        self.curr_obj_mesh_bottom.transform(trans_mtx)

    def get_mesh(self):
        """
        Returns scaled mesh in meters.

        Returns:
            open3d.cuda.pybind.geometry.TriangleMesh: Complete mesh with bottom and articulated top part.
        """
        mesh = self.curr_obj_mesh_top + self.curr_obj_mesh_bottom
        # obj in mm, smpl in m
        trans_mtx = np.eye(4) * (0.001)
        trans_mtx[-1, -1] = 1
        mesh.transform(trans_mtx)
        return mesh