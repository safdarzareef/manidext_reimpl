# Some useful Open3D utils.
import numpy as np
import torch
import math
import open3d as o3d
from utils.utils import getRotation


def pytorch3d_to_open3d(pytorch3d_mesh):
    """
    Convert PyTorch3D mesh to Open3D mesh for visualization.
    """
    # Get vertices and faces from PyTorch3D mesh
    verts = pytorch3d_mesh.verts_packed().detach().cpu().numpy()
    faces = pytorch3d_mesh.faces_packed().detach().cpu().numpy()
    
    # Create Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    
    # Compute vertex normals
    mesh_o3d.compute_vertex_normals()
    
    return mesh_o3d


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


def get_sphere(position, radius, color):
    # modified from ParaHome repo (https://github.com/snuvclab/ParaHome/blob/main/visualize/utils.py)
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius).paint_uniform_color(color)
    sp.translate(position)
    sp.compute_vertex_normals()
    return sp


def get_segment(parent, child, radius, color):
    # modified from ParaHome repo (https://github.com/snuvclab/ParaHome/blob/main/visualize/utils.py)
    v = parent - child
    seg = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(v), resolution=20, split=1).paint_uniform_color(color)
    mat = getRotation(vec1=np.array([0, 0, 1]), vec2=v/np.linalg.norm(v))
    seg.rotate(mat)
    seg.translate((parent+child)/2)
    seg.compute_vertex_normals()
    return seg


def make_cube(w=0.1):
    cube = o3d.geometry.TriangleMesh.create_box(width=w, height=w, depth=w)
    cube.paint_uniform_color([0, 1, 0])  # Green cube
    return cube


def make_arrow(origin, direction, arrow_size=0.2):
    # modified from ParaHome repo (https://github.com/snuvclab/ParaHome/blob/main/visualize/utils.py)
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01,
                                                   cone_radius=0.015,
                                                   cylinder_height=(arrow_size * 0.85),
                                                   cone_height=(arrow_size * 0.15),
                                                   )
    rotation_matrix = getRotation(np.array([0, 0, 1]), direction)
    arrow.rotate(rotation_matrix, center=(0, 0, 0))
    arrow.translate(origin)
    arrow.paint_uniform_color([1, 0, 0])
    return arrow


def get_foot(parent, child, color, width=0.05):
    # modified from ParaHome repo (https://github.com/snuvclab/ParaHome/blob/main/visualize/utils.py)
    v = parent-child
    height = np.linalg.norm(v)
    depth = width/2
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth).paint_uniform_color(color)
    mesh.translate([-width/2, -height/2, -depth/2])
    mat = getRotation(vec1=np.array([0, 1, 0]), vec2=v/height)
    mesh.rotate(mat)
    mesh.translate((parent+child)/2)
    mesh.compute_vertex_normals()
    return mesh


def adjust_playback_speed(start_frame, end_frame, playback_speed):
    # Set playback speed
    speed_up = False
    fps_rate = int(1/playback_speed)

    if playback_speed > 1:
        speed_up = True
        fps_rate = int(playback_speed)

    frame_seq = []
    for fr in range(start_frame, end_frame+1):
        if speed_up:
            if fr % fps_rate == 0:
                frame_seq.append(fr)
        else:
            for _ in range(fps_rate):
                frame_seq.append(fr)

    return frame_seq


class simpleViewer(object):
    # modified from ParaHome repo (https://github.com/snuvclab/ParaHome/blob/main/visualize/utils.py)
    def __init__(self, title, width, height, view=None):
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        self.main_vis = o3d.visualization.O3DVisualizer(title, width, height)
        self.main_vis.show_settings = True
        self.main_vis.show_skybox(False)   
        app.add_window(self.main_vis)

        self.total_frames = 100

        self.orbit_radius = 1.5
        self.orbit_height = 1.75
        self.orbit_angle = math.pi / 4
        self.orbit_cycles = 0.5

        if view is None:
            # width, height = 1600, 800
            view = o3d.camera.PinholeCameraParameters()
            camera_matrix = np.eye(3, dtype=np.float64)
            f = 520
            camera_matrix[0,0] = f
            camera_matrix[1,1] = f
            camera_matrix[0,2] = width/2
            camera_matrix[1,2] = height/2
            view.intrinsic.intrinsic_matrix = camera_matrix
            view.intrinsic.width, view.intrinsic.height = width, height
            self.intrinsic = view.intrinsic

    def export_view(self):
        return self.curview
    
    def setupcamera(self, extrinsic_matrix):
        self.main_vis.setup_camera(self.intrinsic, extrinsic_matrix)

    def tick(self):
        app = o3d.visualization.gui.Application.instance
        tick_return = app.run_one_tick()
        if tick_return:
            self.main_vis.post_redraw()
        return tick_return

    def add_plane(self, resolution=128, bound=100, up_vec='z'):
        def makeGridPlane(bound=100., resolution=128, color = np.array([0.5,0.5,0.5]), up='z'):
            min_bound = np.array([-bound, -bound])
            max_bound = np.array([bound, bound])
            xy_range = np.linspace(min_bound, max_bound, num=resolution)
            grid_points = np.stack(np.meshgrid(*xy_range.T), axis=-1).astype(np.float32) # asd
            if up == 'z':
                grid3d = np.concatenate([grid_points, np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1)], axis=2)
            elif up == 'y':
                grid3d = np.concatenate([grid_points[:,:,0][:,:,None], np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1), grid_points[:,:,1][:,:,None]], axis=2)
            elif up == 'x':
                grid3d = np.concatenate([np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1), grid_points], axis=2)
            else:
                print("Up vector not specified")
                return None
            grid3d = grid3d.reshape((resolution**2,3))
            indices = []
            for y in range(resolution):
                for x in range(resolution):  
                    corner_idx = resolution*y + x 
                    if x + 1 < resolution:
                        indices.append((corner_idx, corner_idx + 1))
                    if y + 1 < resolution:
                        indices.append((corner_idx, corner_idx + resolution))

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(grid3d),
                lines=o3d.utility.Vector2iVector(indices),
            )
            # line_set.colors = o3d.utility.Vector3dVector(colors)  
            line_set.paint_uniform_color(color)
            
            return line_set
        plane = makeGridPlane(bound, resolution, up=up_vec)
        self.main_vis.add_geometry({"name":"floor", "geometry":plane})
        return

    def remove_plane(self):
        self.main_vis.remove_geometry({"name":"floor"})
        return

    def add_geometry(self, geometry:dict):
        self.main_vis.add_geometry(geometry)

    def orbit_camera(self):
        # Update the orbit angle
        self.orbit_angle += (math.pi / self.total_frames) * self.orbit_cycles

        # Calculate new camera position
        x = self.orbit_radius * math.cos(self.orbit_angle)
        y = np.abs(self.orbit_radius * math.sin(self.orbit_angle))
        z = self.orbit_height

        camera_pos = np.array([x, y, z]).reshape(3, 1)
        target = np.array([0, 0, 0]).reshape(3, 1)  # Looking at the origin
        up = np.array([0, 0, 1]).reshape(3, 1)  # Up vector
        self.main_vis.scene.camera.look_at(target, camera_pos, up)


    def write_image(self, imagepath):
        self.main_vis.capture_screen_image(imagepath)


    def transform(self,name, transform_mtx):
        self.main_vis.scene.set_geometry_transform(name, transform_mtx)

    def set_background(self, image):
        self.main_vis.set_background([1, 1, 1, 0], image)

    def remove_geometry(self, geom_name):
        self.main_vis.remove_geometry(geom_name)
    
    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.run()


def select_points(winname='Select points'):
    import cv2
    points = []
    img = np.zeros((1000, 1000, 3))

    def pts_helper(event, x, y, flags, param):
        # Left click to select a point
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(winname, img)

    # Create a window and set the mouse callback
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname, pts_helper)

    # Wait for the user to select 4 points
    while True:
        cv2.imshow(winname, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    points = np.array(points).astype(np.float32)
    points /= np.array([1000, 1000])
    points -= np.array([0.5, 0.5])
    cv2.destroyAllWindows()
    return points


def catmull_rom(percentage, pts_original):
    def catmull_rom_bw_p0_p1(t, p0, p1, p2, p3):
        return 0.5 * ((2 * p1) +
                    (-p0 + p2) * t +
                    (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t ** 2) +
                    (-p0 + 3 * p1 - 3 * p2 + p3) * (t ** 3))
    pts = pts_original.copy()
    pts = np.insert(pts, 0, pts[0], axis=0)
    pts = np.append(pts, [pts[-1]], axis=0)
    assert(len(pts) >= 4)
    num_segments = len(pts) - 3
    segment = min(int(percentage * num_segments), num_segments - 1)
    t = (percentage * num_segments) - segment
    return catmull_rom_bw_p0_p1(t, pts[segment], pts[segment + 1], pts[segment + 2], pts[segment + 3])


def simpleViewerExampleLoop(frame_seq):
    viewer = simpleViewer("Simple Viewer", 1600, 800)
    viewer.total_frames = len(frame_seq)
    global_coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    viewer.add_geometry({"name":"global", "geometry":global_coord})
    viewer.add_plane()

    start_pos = [0, 0, 0]
    end_pos = [1, 1, 1]

    for fr in frame_seq:
        pos =  (end_pos - start_pos) * (fr - frame_seq[0]) / (frame_seq[-1] - frame_seq[0]) + start_pos
        sp = get_sphere(pos, 0.01, [0, 0, 1])
        viewer.remove_geometry("sphere")
        viewer.add_geometry({"name":"sphere", "geometry":sp})

        viewer.orbit_camera()
        viewer.tick()