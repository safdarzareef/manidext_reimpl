# Visualizing the CSE results on MANO hand model
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import tqdm

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from training.mano_cse import MANOSurfaceEmb
from utils.o3d_utils import *


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


def visualize_emb(faces, vertices, embeddings, method='pca'):
    """
    Visualize embeddings projected to colors using PCA
    
    Args:
        vertices: vertex positions (N, 3)
        embeddings: embedding vectors (N, dim)
        method: projection method ('pca' or 'direct')
    """
    # Project embeddings to colors
    colors = project_embeddings_to_colors(embeddings, method)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    # for i in range(vertices.shape[0]):
    #     sp = get_sphere(vertices[i].numpy(), radius=0.001, color=colors[i])
    #     mesh += sp
    
    viewer = simpleViewer("Embedding Visualization", 1600, 800)
    viewer.total_frames = 1000000

    for fr in tqdm.tqdm(range(viewer.total_frames)):
        viewer.remove_geometry(geom_name="vertices")
        viewer.add_geometry({"name":"vertices", "geometry":mesh})
        viewer.tick()

    # o3d.visualization.draw_geometries([mesh])


def visualize_emb_dist(faces, vertices, embeddings, vert_idx):
    """
    Visualize embeddings projected to distances from a given vertex
    
    Args:
        vertices: vertex positions (N, 3)
        embeddings: embedding vectors (N, dim)
        vert_idx: which vertex to measure distances from
    """
    def rgb_color_from_intensity(intensity, colormap='plasma'):
        import matplotlib.pyplot as plt
        intensity = max(0, min(intensity, 1))
        cmap = plt.get_cmap(colormap)
        color = cmap(intensity)[:3]
        return color

    dist = torch.norm(embeddings - embeddings[vert_idx], dim=-1)
    dist = dist.numpy()
    dist = (dist - dist.min()) / (dist.max() - dist.min())
    dist_intensity = 1 - dist

    colors = np.zeros((vertices.shape[0], 3))
    for i in range(vertices.shape[0]):
        color = rgb_color_from_intensity(dist_intensity[i])
        colors[i] = color

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    
    viewer = simpleViewer("Embedding Visualization", 1600, 800)
    viewer.total_frames = 100000

    for fr in tqdm.tqdm(range(viewer.total_frames)):
        viewer.remove_geometry(geom_name="vertices")
        viewer.add_geometry({"name":"vertices", "geometry":mesh})
        viewer.tick()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=20, help="Dimension of embeddings")
    parser.add_argument("--rhand", action="store_true", help="Use right hand model else left hand")
    parser.add_argument("--vis_emb", action="store_true", help="Visualize embeddings")
    parser.add_argument("--vis_dist", action="store_true", help="Visualize distances from a vertex")
    parser.add_argument("--vert_idx", type=int, default=-1, help="Vertex index for distance visualization")
    args = parser.parse_args()

    mano_cse = MANOSurfaceEmb(is_rhand=args.rhand, emb_dim=args.emb_dim)
    embeddings = mano_cse.emb.detach().cpu()
    faces = mano_cse.faces
    vertices = mano_cse.default_vertices

    if args.vis_emb:
        visualize_emb(faces, vertices, embeddings.numpy(), method='pca')
    if args.vis_dist:
        vert_idx = np.random.randint(0, 778-1) if args.vert_idx == -1 else args.vert_idx
        visualize_emb_dist(faces, vertices, embeddings, vert_idx)