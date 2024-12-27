import numpy as np
import torch
import torch.nn.functional as F
import math

from scipy.spatial.transform import Rotation as R


def transform_points(points, transformation_matrix):
    if torch.is_tensor(points):
        pts_shape = points.shape
        points = points.view(-1, 3)
        homogeneous_points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
        if not torch.is_tensor(transformation_matrix):
            transformation_matrix = torch.tensor(transformation_matrix, dtype=points.dtype)
        transformation_matrix = transformation_matrix.to(points.device)
        transformed_points = homogeneous_points @ transformation_matrix.T
        return transformed_points[:, :3].view(pts_shape)
    else:
        pts_shape = points.shape
        points = points.reshape(-1, 3)
        homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = homogeneous_points @ transformation_matrix.T
        return transformed_points[:, :3].reshape(pts_shape)
    
def l2_dist(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def normalize(vec):
    return vec / np.linalg.norm(vec)


def getRotation(vec1, vec2):
    # from ParaHome repo (https://github.com/snuvclab/ParaHome/blob/main/visualize/utils.py)
    vec1 = vec1/np.sqrt(vec1[0]**2+vec1[1]**2+vec1[2]**2)
    vec2 = vec2/np.sqrt(vec2[0]**2+vec2[1]**2+vec2[2]**2)

    n = np.cross(vec1, vec2)

    v_s = np.sqrt(n[0]**2+n[1]**2+n[2]**2)
    v_c = np.dot(vec1, vec2)
    skew = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    if v_s != 0:
        rotmat = np.eye(3) + skew+ skew@skew*((1-v_c)/(v_s**2))
    else:
        rotmat = np.eye(3) + skew
    return rotmat


def rotation_matrix_from_axis_angle(axis, angle_degrees):
    # Normalize the axis vector
    axis = np.array(axis) / np.linalg.norm(axis)
    
    # Create a Rotation object
    r = R.from_rotvec(np.radians(angle_degrees) * axis).as_matrix()
    
    # Get the rotation matrix
    return r


def find_nearest_vertices(x_vertices, y_vertices):
    if x_vertices.shape == y_vertices.shape:
        raise ValueError("Both sets of vertices must have the same dimensionality")
    
    if not torch.is_tensor(x_vertices):
        x_vertices = torch.tensor(np.array(x_vertices))
    if not torch.is_tensor(y_vertices):
        y_vertices = torch.tensor(np.array(y_vertices))

    x_expanded = x_vertices.unsqueeze(-2)  # Shape: (m, 1, d)
    y_expanded = y_vertices.unsqueeze(-3)  # Shape: (1, n, d)
    distances = torch.sum((x_expanded - y_expanded) ** 2, dim=-1)  # Shape: (m, n)
    
    # Find minimum distances and their indices
    min_distances, nearest_indices = torch.min(distances, dim=-1)  # Shape: (m,), (m,)
    nearest_indices = list(nearest_indices.numpy())
    min_distances = min_distances.numpy()

    # Return both indices and square root of minimum distances
    return nearest_indices, np.sqrt(min_distances)