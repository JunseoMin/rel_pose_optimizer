import open3d as o3d
import numpy as np

def visualize_pose(relpose_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    world_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    vis.add_geometry(world_mesh)

    obj_frames = []
    traj_points = []

    for T_rel in relpose_list:
        T_current = T_rel

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(T_current)
        obj_frames.append(frame)

        traj_points.append(T_current[:3, 3])

    for frame in obj_frames:
        vis.add_geometry(frame)

    if len(traj_points) > 1:
        traj_line_set = o3d.geometry.LineSet()
        traj_points = np.array(traj_points)

        traj_lines = [[i, len(traj_points) - 1] for i in range(len(traj_points) - 1)]
        
        traj_line_set.points = o3d.utility.Vector3dVector(traj_points)
        traj_line_set.lines = o3d.utility.Vector2iVector(traj_lines)
        traj_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in traj_lines])  # 빨간색

        vis.add_geometry(traj_line_set)

    vis.run()
    vis.destroy_window()
