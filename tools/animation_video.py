"""
# Created: 2023-11-29 21:22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: view scene flow dataset after preprocess.
"""

import numpy as np
import fire, time
import open3d as o3d
import os, sys, glob, json
from scipy.interpolate import CubicSpline
import cv2
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, egopts_mask
from src.utils.o3d_view import MyVisualizer, hex_to_rgb, color_map

# color_map_hex = ['#1f78b4']
# color_map = [hex_to_rgb(color) for color in color_map_hex]
color_map_hex = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
# # color_map_hex = ['#b2df8a', '#1f78b4']
color_map = [hex_to_rgb(color) for color in color_map_hex]


def interpolate_trajectory(traj, sample_step=10):
    '''
    Interpolate a camera trajectory using cubic spline interpolation.
    Args:
        traj: a list of dictionaries, each containing 'front', 'lookat', 'up', 'zoom' keys
        sample_step: the number of samples to take between each pair of keyframes
    Returns:
        a expands list of dictionaries, each containing 'front', 'lookat', 'up', 'zoom' keys
    '''
    fronts, lookats, ups, zooms = [], [], [], []
    for frame in traj:
        fronts.append(frame['front'])
        lookats.append(frame['lookat'])
        ups.append(frame['up'])
        zooms.append(frame['zoom'])
    
    # Time vector corresponding to each keyframe
    t = np.arange(len(traj))
   
    # Interpolation function for each component
    front_spline = CubicSpline(t, np.array(fronts), bc_type='clamped')
    lookat_spline = CubicSpline(t, np.array(lookats), bc_type='clamped')
    up_spline = CubicSpline(t, np.array(ups), bc_type='clamped')
    zoom_spline = CubicSpline(t, np.array(zooms), bc_type='clamped')
    
    # Generate interpolated values
    t_new = np.linspace(0, t[-1], num=len(traj) * sample_step - (sample_step - 1))
    # print("sample_step: ", sample_step, "len(t_new): ", len(t_new), "num: ", len(traj) * sample_step - (sample_step - 1))
    interpolated_traj = []
    for ti in t_new:
        interpolated_traj.append({
            'front': front_spline(ti).tolist(),
            'lookat': lookat_spline(ti).tolist(),
            'up': up_spline(ti).tolist(),
            'zoom': zoom_spline(ti).tolist()
        })
    
    return interpolated_traj

def custom_draw_geometry_with_camera_trajectory(mesh, trajs, img_folder='logs/imgs', point_size=3.0):
    # os.makedirs(img_folder, exist_ok=True)
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.traj = trajs
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    img_array = []
    def move_forward(vis):
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            pass
            image = vis.capture_screen_float_buffer(False)
            image = (np.clip(np.asarray(image) * 255, 0, 255)).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_array.append(image)
            # print("Capture image {:05d}".format(glb.index))
            # vis.capture_screen_image(f"{img_folder}/{glb.index:05d}.png", False)
        glb.index = glb.index + 1
        if glb.index < len(glb.traj):
            # ctr.change_field_of_view(glb.traj[glb.index]['field_of_view'])
            ctr.set_front(glb.traj[glb.index]['front'])
            ctr.set_lookat(glb.traj[glb.index]['lookat'])
            ctr.set_up(glb.traj[glb.index]['up'])
            ctr.set_zoom(glb.traj[glb.index]['zoom'])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
            custom_draw_geometry_with_camera_trajectory.vis.destroy_window()
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    opt = vis.get_render_option()
    opt.point_size = point_size
    vis.add_geometry(mesh)
    # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2))
    vis.register_animation_callback(move_forward)
    vis.run()
    return img_array

def animation_video(
    data_dir: str ="/home/kin/data/Scania/preprocess/val",
    flow_mode: str = "raw", # "flow", "flow_est"
    start_id: int = 200,
    point_size: float = 4.0,
    ins_id = [1, 3],
):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True)
    save_img_folder = f"{BASE_DIR}/logs/imgs/raw_{start_id}_{ins_id[0]}"

    data = dataset[start_id]
    pc0 = data['pc0']
    gm0 = data['ground_mask0'] | ~egopts_mask(pc0)
    pc0 = pc0[~gm0]
    pose0 = data['pose0']
    pose1 = data['pose1']
    ego_pose = np.linalg.inv(pose1) @ pose0
    pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]

    lidar_id = data['lidar_id'][~gm0]
    flow = pose_flow if flow_mode == 'raw' else data[flow_mode][~gm0]

    # mask = np.zeros_like(gm0[:len(pc0)], dtype=bool)
    # for i in ins_id:
    #     mask[data['flow_instance_id'][~gm0]==i] = True
    mask = np.ones_like(gm0[:len(pc0)], dtype=bool)

    vis_pc_ = pc0[mask][:,:3]
    vis_id = lidar_id[mask]
    vis_flow = flow[mask] - pose_flow[mask]
    # vis_flow[np.linalg.norm(vis_flow, axis=1).mean() < 1.4] = 0
    vis_dt0 = data['dt0'][~gm0][mask]
    vis_dt0 = max(vis_dt0) - vis_dt0
    vis_pc = vis_pc_ + (vis_flow/0.1) * vis_dt0[:, None] 
    # zod 2Hz
    # vis_pc = vis_pc_ + (vis_flow/0.45) * vis_dt0[:, None] 
    pcd = o3d.geometry.PointCloud()
    for lidar_id in np.unique(vis_id):
        mask = vis_id == lidar_id
        single_pcd = o3d.geometry.PointCloud()
        single_pcd.points = o3d.utility.Vector3dVector(vis_pc[mask])
        single_pcd.paint_uniform_color(color_map[lidar_id % len(color_map)])
        pcd += single_pcd
    # add raw for view same...
    # pcd_raw = o3d.geometry.PointCloud()
    # pcd_raw.points = o3d.utility.Vector3dVector(vis_pc_)
    # pcd_raw.paint_uniform_color([1, 1, 1])
    # pcd += pcd_raw

    trajs = []
    load_logtraj = False
    if os.path.isdir(save_img_folder) and (not os.path.exists(f'{save_img_folder}/trajs.log')):
        json_files = sorted(glob.glob('{}/*.json'.format(save_img_folder)))
        for file in json_files:
            with open(file, 'r') as f:
                trajs.append(json.loads(f.read())['trajectory'][0])
    elif save_img_folder.endswith('.log') or os.path.exists(f'{save_img_folder}/trajs.log'):
        file = f'{save_img_folder}/trajs.log' if os.path.isdir(save_img_folder) else save_img_folder
        print('Loading trajectory from old log file... If you want to re-interpolate, please set --overwrite True.')
        with open(file, 'r') as f:
            trajs = json.loads(f.read())['trajectory']
        load_logtraj = True
    if not load_logtraj:
        trajs = interpolate_trajectory(trajs, 100)
        with open(f'{save_img_folder}/trajs.log', 'w') as f:
            f.write(json.dumps({'trajectory': trajs}))
        print('Interpolated trajectory saved to trajs.log inside:', save_img_folder, "You will use this file next time by default.")

    save_folder = f"{BASE_DIR}/logs/video"
    os.makedirs(save_folder, exist_ok=True)
    save_video = save_folder+f"/{flow_mode}_{start_id}_{ins_id[0]}"
    img_arrays = custom_draw_geometry_with_camera_trajectory(pcd, trajs, img_folder=save_video, point_size=point_size)

    view_single_img_t = 1/len(trajs) * 5
    height, width = img_arrays[0].shape[:2]
    out = cv2.VideoWriter(f'{save_video}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1/view_single_img_t, (width, height))
    for i in tqdm(range(len(img_arrays)), ncols=100):
        out.write(img_arrays[i])
    out.release()

def lidar_frames(data):
    SensorsCenter = data['SensorsCenter']
    # print(data['other_infos'])
    lidar_frames = []
    for i in range(SensorsCenter.shape[0]):
        lidar_frame = np.eye(4)
        lidar_frame[:3, 3] = SensorsCenter[i]
        lidar_frames.append(lidar_frame)
    o3d_frame = []
    for coordinate_frame in lidar_frames:
        o3d_frame.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).transform(coordinate_frame))
    return o3d_frame

def real_vis(
    data_dir: str ="/home/kin/data/Scania/preprocess/val",
    flow_mode: str = "himu_seflowpp", # "flow", "flow_est"
    start_id: int = 200,
    point_size: float = 4.0,
    ins_id = [1, 3],
):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True)
    save_img_folder = f"{BASE_DIR}/logs/autovideos/av2/{flow_mode}"
    o3d_vis = MyVisualizer(view_file="/home/kin/workspace/himu/assets/view/video_view_scania.json", window_title=f"view {'ground truth flow' if flow_mode == 'flow' else f'{flow_mode} flow'}, `SPACE` start/stop", \
                           save_folder=save_img_folder)

    opt = o3d_vis.vis.get_render_option()
    opt.point_size = point_size
    for data_id in (pbar := tqdm(range(start_id, len(dataset)))):
        data = dataset[data_id]
        pbar.set_description(f"id: {data_id}, logid: {data['scene_id']}, timestamp: {data['timestamp']}")
        pc0 = data['pc0']
        gm0 = data['ground_mask0'] | ~egopts_mask(pc0)
        pc0 = pc0[~gm0]
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]

        lidar_id = data['lidar_id'][~gm0]
        flow = pose_flow if flow_mode == 'raw' else data[flow_mode][~gm0]

        # mask = np.zeros_like(gm0[:len(pc0)], dtype=bool)
        # for i in [1, 3]:
        #     mask[data['flow_instance_id'][~gm0]==i] = True
        mask = np.ones_like(gm0[:len(pc0)], dtype=bool)

        vis_pc_ = pc0[mask][:,:3]
        vis_id = lidar_id[mask]
        vis_flow = flow[mask] - pose_flow[mask]
        vis_dt0 = data['dt0'][~gm0][mask]
        vis_dt0 = vis_dt0 - min(vis_dt0)
        vis_dt0 = max(vis_dt0) - vis_dt0
        
        # normal 10Hz
        vis_pc = vis_pc_ + (vis_flow/0.1) * vis_dt0[:, None]

        # zod 2Hz? check again later
        # vis_pc = vis_pc_ + (vis_flow/0.35) * vis_dt0[:, None] 
        pcd = o3d.geometry.PointCloud()
        for lidar_id_ in np.unique(vis_id):
            mask = vis_id == lidar_id_
            single_pcd = o3d.geometry.PointCloud()
            single_pcd.points = o3d.utility.Vector3dVector(vis_pc[mask])
            single_pcd.paint_uniform_color(color_map[(lidar_id_) % len(color_map)])
            # single_pcd.paint_uniform_color(color_map[0])
            pcd += single_pcd

        # bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-10.0, -2.760004/2, 0], max_bound=np.array([2.760004, 2.760004/2, 5]))
        # if True:
        #     o3d_frame = lidar_frames(data)
        #     o3d_vis.update([pcd, *o3d_frame]) # , bbx
        # else:
        o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)]) # , bbx
        # o3d_vis.update([pcd])
        # o3d_vis.save_screen(figure_name=data_id)
        # if data_id > 580:
        #     break
def save_animation_traj(
    data_dir: str ="/home/kin/data/Scania/preprocess/val",
    flow_mode: str = "raw", # "flow", "flow_est"
    start_id: int = 270,
    point_size: float = 4.0,
    ins_id = [1, 3],
):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True)
    save_img_folder = f"{BASE_DIR}/logs/imgs/{flow_mode}_{start_id}_{ins_id[0]}"
    o3d_vis = MyVisualizer(view_file=None, window_title=f"view {'ground truth flow' if flow_mode == 'flow' else f'{flow_mode} flow'}, `SPACE` start/stop", \
                           save_folder=save_img_folder)

    opt = o3d_vis.vis.get_render_option()
    opt.point_size = point_size

    data = dataset[start_id]
    pc0 = data['pc0']
    gm0 = data['ground_mask0'] | ~egopts_mask(pc0)
    pc0 = pc0[~gm0]
    pose0 = data['pose0']
    pose1 = data['pose1']
    ego_pose = np.linalg.inv(pose1) @ pose0
    pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]

    lidar_id = data['lidar_id'][~gm0]
    flow = pose_flow if flow_mode == 'raw' else data[flow_mode][~gm0]

    # mask = np.zeros_like(gm0[:len(pc0)], dtype=bool)
    # for i in [1, 3]:
    #     mask[data['flow_instance_id'][~gm0]==i] = True
    mask = np.ones_like(gm0[:len(pc0)], dtype=bool)

    vis_pc_ = pc0[mask][:,:3]
    vis_id = lidar_id[mask]
    vis_flow = flow[mask] - pose_flow[mask]
    vis_dt0 = data['dt0'][~gm0][mask]
    vis_dt0 = vis_dt0 - min(vis_dt0)
    vis_dt0 = max(vis_dt0) - vis_dt0
    
    # normal 10Hz
    vis_pc = vis_pc_ + (vis_flow/0.1) * vis_dt0[:, None]

    # zod 2Hz? check again later
    # vis_pc = vis_pc_ + (vis_flow/0.35) * vis_dt0[:, None] 
    pcd = o3d.geometry.PointCloud()
    for lidar_id_ in np.unique(vis_id):
        mask = vis_id == lidar_id_
        single_pcd = o3d.geometry.PointCloud()
        single_pcd.points = o3d.utility.Vector3dVector(vis_pc[mask])
        single_pcd.paint_uniform_color(color_map[(lidar_id_+1) % len(color_map)])
        pcd += single_pcd

    # bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-10.0, -2.760004/2, 0], max_bound=np.array([2.760004, 2.760004/2, 5]))
    o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)]) # , bbx
    
if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(real_vis)
    # fire.Fire(save_animation_traj)
    # fire.Fire(animation_video)
    # for flow_name in ['raw', 'fastflow3d_best', 'deflow_best', 'nsfp', 'fastnsf10', 'icpflow', 'himu_seflow', 'himu_seflowpp']:
    # for flow_name in ['raw', 'himu_seflowpp']:
    #     animation_video(flow_mode=flow_name)
    # for flow_name in ['raw', 'seflow_best']:
    # for flow_name in ['raw', 'fastnsf']:
    #     animation_video(flow_mode=flow_name)
    print(f"Time used: {time.time() - start_time:.2f} s")