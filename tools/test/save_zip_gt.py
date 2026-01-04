"""
# Created: 2024-04-15 17:32
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)

# Description: 
# Save the ground truth compensation distance into feather files inside a zip file for 
# afterward easy evaluation on the public benchmark.


PUT THIS FILE INTO HIMO FOLDER AND RUN.
"""

import fire, time, json
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
NDArrayFloat = npt.NDArray[np.float64]

from typing import Tuple
from pathlib import Path
import pandas as pd
from zipfile import ZipFile
from io import BytesIO
import time, os, sys

BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), 'OpenSceneFlow' ))
sys.path.append(BASE_DIR)
from src.dataset import HDF5Dataset
from src.utils.av2_eval import CLOSE_DISTANCE_THRESHOLD
from utils import check_valid, ego_pts_mask, flow2compDis, refine_pts


def read_output_zip(
    zip_path: str,
    sweep_uuid: Tuple[str, int],
) -> np.ndarray:
    """Read compensation distance predictions from a zip file.

    Args:
        zip_path: Path to the zip file containing predictions.
        sweep_uuid: Identifier of the sweep being predicted (log_id, timestamp_ns).

    Returns:
        compensation_dis: (N, 3) compensation distance predictions.
    """
    with ZipFile(zip_path, 'r') as myzip:
        feather_path = f"{sweep_uuid[0]}/{sweep_uuid[1]}.feather"
        with myzip.open(feather_path) as f:
            df = pd.read_feather(BytesIO(f.read()))
    
    compensation_dis = np.stack([
        df['comp_dis_x_m'].values.astype(np.float32),
        df['comp_dis_y_m'].values.astype(np.float32),
        df['comp_dis_z_m'].values.astype(np.float32),
    ], axis=1)
    
    # If category/instance columns are present, they will be available in the DataFrame
    eval_mask = df['eval_mask'].values.astype(bool)
    flow_category = df['flow_category_indices'].values.astype(np.uint8) if 'flow_category_indices' in df.columns else None
    flow_instance = df['flow_instance_id'].values.astype(np.uint32) if 'flow_instance_id' in df.columns else None
    return compensation_dis, eval_mask, flow_category, flow_instance
    
def write_output_file(
    compensation_dis: NDArrayFloat,
    sweep_uuid: Tuple[str, int],
    output_dir: Path,
    eval_mask: NDArrayFloat,
    flow_category_indices: np.ndarray = None,
    flow_instance_id: np.ndarray = None,
    gt_flow_norm: np.ndarray = None,
    pc0: np.ndarray = None,
) -> None:
    """Write an output predictions file in the correct format for submission.

    Args:
        compensation_dis: (N,3) compensation_dis predictions.
        sweep_uuid: Identifier of the sweep being predicted (log_id, timestamp_ns).
        output_dir: Top level directory containing all predictions.
        pc0: (N,3) original point cloud coordinates (needed for Chamfer calculation).
    """
    output_log_dir = output_dir / sweep_uuid[0]
    output_log_dir.mkdir(exist_ok=True, parents=True)
    compensation_dis_x_m = compensation_dis[:, 0].astype(np.float32)
    compensation_dis_y_m = compensation_dis[:, 1].astype(np.float32)
    compensation_dis_z_m = compensation_dis[:, 2].astype(np.float32)

    # Build the output DataFrame; include category/instance if provided
    data_dict = {
        "comp_dis_x_m": compensation_dis_x_m,
        "comp_dis_y_m": compensation_dis_y_m,
        "comp_dis_z_m": compensation_dis_z_m,
        "eval_mask": eval_mask.astype(np.uint8),
    }
    if flow_category_indices is not None:
        data_dict['flow_category_indices'] = flow_category_indices.astype(np.uint8)
    if flow_instance_id is not None:
        data_dict['flow_instance_id'] = flow_instance_id.astype(np.uint32)
    if gt_flow_norm is not None:
        data_dict['gt_flow_norm'] = gt_flow_norm.astype(np.float32)
    # Save pc0 for Chamfer calculation (GT only)
    if pc0 is not None:
        data_dict['pc0_x'] = pc0[:, 0].astype(np.float32)
        data_dict['pc0_y'] = pc0[:, 1].astype(np.float32)
        data_dict['pc0_z'] = pc0[:, 2].astype(np.float32)

    output = pd.DataFrame(data_dict)
    output.to_feather(output_log_dir / f"{sweep_uuid[1]}.feather")


def zip_res(res_folder, output_file="submit.zip"):
    all_scenes = [f for f in os.listdir(res_folder) if os.path.isdir(os.path.join(res_folder, f))]
    with ZipFile(output_file, "w") as myzip:
        for scene in all_scenes:
            scene_folder = os.path.join(res_folder, scene)
            # only directory
            all_logs = [f for f in os.listdir(scene_folder) if os.path.isfile(os.path.join(scene_folder, f)) and f.endswith('.feather')]
            for log in all_logs:
                file_path = os.path.join(scene, log)
                myzip.write(os.path.join(res_folder, file_path), arcname=file_path)
    # remove the folder after zipping
    for scene in all_scenes:
        scene_folder = os.path.join(res_folder, scene)
        os.system(f"rm -rf {scene_folder}")
    print(f"Zipped results to {res_folder} into {output_file}. Submit your result by uploading this zip file.")
    # print()
    return output_file

def main(
    # data_dir: str ="/home/kin/data/Scania/preprocess/val",
    data_dir: str ="/home/kin/data/av2/h5py/sensor/himo/demo",
    output_dir: str = "/home/kin/data/av2/h5py/sensor/himo/results",
    res_name: str = "flow"
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    data_name, _ = check_valid(str(data_dir), res_name, None)

    dataset = HDF5Dataset(data_dir, vis_name=res_name, eval=True)
    for data_id in tqdm(range(0, len(dataset)), ncols=120, desc=f"Extracting {res_name} from {data_dir}"):
        data = dataset[data_id]
        pc0, pose0, pose1 = data['pc0'], data['pose0'], data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        pc_dis = np.linalg.norm(pc0[:, :2], axis=1)
        dis_mask = pc_dis <= CLOSE_DISTANCE_THRESHOLD
        notgm_mask = ~data['gm0']

        # scania
        if data_name == "scania":
            mask_eval = dis_mask & data['flow_is_valid'] & notgm_mask & ego_pts_mask(pc0)
        else:
            mask_eval = dis_mask & notgm_mask & ego_pts_mask(pc0, min_bound=[-1.5, -1.5, -2.0], max_bound=[1.5, 1.5, 2.0])
        
        try:
            est_flow = np.zeros_like(pose_flow) if res_name == "raw" else (data[res_name] - pose_flow)
        except:
            print(f"Warning: {data['scene_id']} {data['timestamp']} has no result for {res_name}, set zero flow.")
            est_flow = np.zeros_like(pose_flow)

        # NOTE: we compensated to the latest observation. dts: (N,1) Nanosecond offsets _from_ the start of the sweep.
        # NOTE: we compensated to the latest observation. dts: (N,1) Nanosecond offsets _from_ the start of the sweep.
        dt0 = max(data['lidar_dt']) - data['lidar_dt']
        
        # GT flow and GT compensation distance
        gt_flow = data['flow'] - pose_flow  # GT flow in ego-motion compensated frame
        gt_comp_dis = flow2compDis(gt_flow, dt0, sensor_dt=0.1)  # GT compensation distance
        gt_flow_norm = np.linalg.norm(gt_flow, axis=1).astype(np.float32)

        # Attach category/instance arrays so the saved feather contains labels for GT
        flow_cat = data['flow_category_indices'] if 'flow_category_indices' in data else None
        flow_inst = data['flow_instance_id'] if 'flow_instance_id' in data else None
        
        # Save GT compensation distance (for GT zip file)
        write_output_file(gt_comp_dis, (data['scene_id'], str(data['timestamp'])), output_dir, mask_eval, 
                          flow_category_indices=flow_cat, flow_instance_id=flow_inst, 
                          gt_flow_norm=gt_flow_norm, pc0=pc0[:, :3])

    zip_res(output_dir, output_file=f"{output_dir}/{res_name}-submit.zip")

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")