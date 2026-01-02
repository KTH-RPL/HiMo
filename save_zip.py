"""
# Created: 2024-04-15 17:32
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)

# Description: save the distance to based on the data file, need save result first. 
#              Run 4_vis.py to produce output append in existing .h5 file.
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
    
    return compensation_dis
    
def write_output_file(
    compensation_dis: NDArrayFloat,
    sweep_uuid: Tuple[str, int],
    output_dir: Path,
) -> None:
    """Write an output predictions file in the correct format for submission.

    Args:
        compensation_dis: (N,3) compensation_dis predictions.
        sweep_uuid: Identifier of the sweep being predicted (log_id, timestamp_ns).
        output_dir: Top level directory containing all predictions.
    """
    output_log_dir = output_dir / sweep_uuid[0]
    output_log_dir.mkdir(exist_ok=True, parents=True)
    compensation_dis_x_m = compensation_dis[:, 0].astype(np.float32)
    compensation_dis_y_m = compensation_dis[:, 1].astype(np.float32)
    compensation_dis_z_m = compensation_dis[:, 2].astype(np.float32)

    output = pd.DataFrame(
        {
            "comp_dis_x_m": compensation_dis_x_m,
            "comp_dis_y_m": compensation_dis_y_m,
            "comp_dis_z_m": compensation_dis_z_m,
        }
    )
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
    res_name: str = "seflowpp_best"
):
    data_dir = Path(data_dir)
    output_dir = data_dir / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset = HDF5Dataset(data_dir, vis_name=res_name, eval=True)
    for data_id in tqdm(range(0, len(dataset)), ncols=120, desc=f"Extracting {res_name} from {data_dir}"):
        data = dataset[data_id]
        pc0, pose0, pose1 = data['pc0'], data['pose0'], data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        est_flow = np.zeros_like(pose_flow) if res_name == "raw" else (data[res_name] - pose_flow)

        # NOTE: we compensated to the latest observation. dts: (N,1) Nanosecond offsets _from_ the start of the sweep.
        dt0 = max(data['lidar_dt']) - data['lidar_dt']
        comp_dis = flow2compDis(est_flow, dt0, sensor_dt=0.1) # [N, 3] each point's compensation distance

        write_output_file(comp_dis, (data['scene_id'], str(data['timestamp'])), output_dir)

    zip_res(output_dir, output_file=f"{output_dir}/{res_name}-submit.zip")

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")