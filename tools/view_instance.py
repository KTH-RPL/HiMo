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
from tqdm import tqdm
import open3d as o3d
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

from src.utils.mics import HDF5Data, egopts_mask
from src.utils.o3d_view import MyVisualizer, hex_to_rgb
color_map_hex = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
color_map = [hex_to_rgb(color) for color in color_map_hex]

VIEW_FILE = f"{BASE_DIR}/assets/view/scania.json"

def print_score(
    pc_data: np.ndarray,
    gt_flow: np.ndarray,
    est_flow: np.ndarray,
    delta_t: np.ndarray,
    flow_mode: str = "raw",
):
    est_pc = pc_data + (est_flow/0.1) * delta_t[:, None] 
    gt_pc = pc_data + (gt_flow/0.1) * delta_t[:, None]
    distance_matrix_12 = np.linalg.norm(est_pc[:, None] - gt_pc, axis=2)
    distance_matrix_21 = np.linalg.norm(gt_pc[:, None] - est_pc, axis=2)
    cham = (np.nanmean(np.min(distance_matrix_12, axis=1)) + np.nanmean(np.min(distance_matrix_21, axis=1))) / 2
    mpe = np.linalg.norm(est_flow - gt_flow, axis=1).mean()
    print(f"chamfer distance: {cham:.4f}, mean point error: {mpe:.4f}, gt speed: {np.linalg.norm(gt_flow, axis=1).mean():.3f}, est speed: {max(np.linalg.norm(est_flow, axis=1)):.3f}")
    return cham, mpe

def print_refine_ins(
    data_dir: str ="/home/kin/data/Scania/preprocess/val",
    flow_mode: str = "flow", # "flow", "flow_est"
    start_id: int = 270,
    point_size: float = 4.0,
    ins_id = [1, 3],
):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True)
    for data_id in range(start_id, len(dataset)):
        data = dataset[data_id]
        pc0 = data['pc0']
        gm0 = data['ground_mask0']
        pc0 = pc0[~gm0]
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]

        flow = pose_flow if flow_mode == 'raw' else data[flow_mode][~gm0]

        mpes = []
        cdes = []
        num_pts = []
        for i in ins_id:
            mask = np.zeros_like(gm0[:len(pc0)], dtype=bool)
            mask[data['flow_instance_id'][~gm0]==i] = True
            vis_pc = pc0[mask][:,:3]
            vis_flow = flow[mask] - pose_flow[mask]
            vis_dt0 = data['dt0'][~gm0][mask]
            cde, mpe  = print_score(vis_pc, data['flow'][~gm0][mask] - pose_flow[mask], vis_flow, vis_dt0, flow_mode)
            mpes.append(mpe)
            cdes.append(cde)
            num_pts.append(vis_pc.shape[0])
            # print(f"ins_id: {i}, chamfer distance: {cde:.4f}, mean point error: {mpe:.4f}, speed: {np.linalg.norm(vis_flow, axis=1).mean():.4f}")
        mpe = np.average(mpes, weights=num_pts)
        cde = np.average(cdes, weights=num_pts)


        print(f"\n {flow_mode} flow:")
        print(f"chamfer distance: {cde:.4f}")
        print(f"mean point error: {mpe:.4f}")
        break

def vis_refine_ins(
    data_dir: str ="/home/kin/data/Scania/preprocess/val",
    flow_mode: str = "raw",
    start_id: int = 109,
    point_size: float = 4.0,
):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True, eval=False)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if flow_mode == 'flow' else f'{flow_mode} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    opt.point_size = point_size

    for data_id in (pbar := tqdm(range(0, len(dataset)))):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        # for easy stop and jump to any id
        if data_id < start_id and start_id != -1:
            continue

        pc0 = data['pc0']
        gm0 = data['ground_mask0'] | ~egopts_mask(pc0)
        pc0 = pc0[~gm0]
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]

        lidar_id = data['lidar_id'][~gm0]
        flow = pose_flow if flow_mode == 'raw' else data[flow_mode][~gm0]

        # option A: view selected instance only
        mask = np.zeros_like(gm0[:len(pc0)], dtype=bool)
        for i in [8]:
        # for i in range(13, 20):
            mask[data['flow_instance_id'][~gm0]==i] = True
        # print('category:', np.unique(data['flow_category_indices'][~gm0][mask]))
        # # option B: to view all pts...
        # mask = np.ones_like(gm0[:len(pc0)], dtype=bool)

        vis_pc = pc0[mask][:,:3]
        vis_id = lidar_id[mask]
        vis_flow = flow[mask] - pose_flow[mask]
        vis_dt0 = data['dt0'][~gm0][mask]
        vis_dt0 = max(vis_dt0) - vis_dt0

        print_score(vis_pc, data['flow'][~gm0][mask] - pose_flow[mask], vis_flow, vis_dt0, flow_mode)
        # break

        vis_pc = vis_pc + (vis_flow/0.1) * vis_dt0[:, None] 
        pcd = o3d.geometry.PointCloud()
        for lidar_id in np.unique(vis_id):
            mask = vis_id == lidar_id
            single_pcd = o3d.geometry.PointCloud()
            single_pcd.points = o3d.utility.Vector3dVector(vis_pc[mask])
            single_pcd.paint_uniform_color(color_map[lidar_id % len(color_map)])
            pcd += single_pcd
        # pcd.points = o3d.utility.Vector3dVector(vis_pc[:,:3])
        # pcd.colors = o3d.utility.Vector3dVector(np.ones((vis_pc.shape[0], 3)))
        # for i in range(vis_pc.shape[0]):
        #     pcd.colors[i] = color_map[vis_id[i] % len(color_map)]

        # bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-10.0, -2.760004/2, 0], max_bound=np.array([2.760004, 2.760004/2, 5]))
        o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)]) # , bbx

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(vis_refine_ins)
    # for flow_name in ['raw', 'fastflow3d_best', 'deflow_best', 'nsfp', 'fastnsf10', 'icpflow', 'himu_seflow', 'himu_seflowpp']:
    # for flow_name in ['raw', 'deflow_best', 'nsfp', 'fastnsf10', 'himu_seflow', 'himu_seflowpp']:
        # print_refine_ins(flow_mode=flow_name, start_id=270, ins_id=[3])
    # print_refine_ins(flow_mode='flow', start_id=270, ins_id=[1,3])
    print(f"Time used: {time.time() - start_time:.2f} s")