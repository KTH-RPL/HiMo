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
import time, os, sys

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
    compensation_dis_x_m = compensation_dis[:, 0].astype(np.float16)
    compensation_dis_y_m = compensation_dis[:, 1].astype(np.float16)
    compensation_dis_z_m = compensation_dis[:, 2].astype(np.float16)

    output = pd.DataFrame(
        {
            "comp_dis_x_m": compensation_dis_x_m,
            "comp_dis_y_m": compensation_dis_y_m,
            "comp_dis_z_m": compensation_dis_z_m,
        }
    )
    output.to_feather(output_log_dir / f"{sweep_uuid[1]}.feather")


def zip_res(res_folder, output_file="submit.zip"):
    all_scenes = os.listdir(res_folder)
    start_time = time.time()

    with ZipFile(output_file, "w") as myzip:
        for scene in all_scenes:
            scene_folder = os.path.join(res_folder, scene)
            all_logs = os.listdir(scene_folder)
            for log in all_logs:
                if not log.endswith(".feather"):
                    continue
                file_path = os.path.join(scene, log)
                myzip.write(os.path.join(res_folder, file_path), arcname=file_path)

    print(f"Time cost: {time.time()-start_time:.2f}s, check the zip file: {output_file}")
    return output_file

def main(
    # data_dir: str ="/home/kin/data/Scania/preprocess/val",
    data_dir: str ="/home/kin/data/av2/h5py/sensor/himo_val",
    output_dir: str = "./output/val",
    res_name: str = "seflowpp"
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # PLACEHOLDER for potential method with ds directly
    if res_name in ['direct_ds']:
        pass
    else:
        pass
        # haven't finished yet here.
        # write_output_file(compensation_dis, sweep_uuid, output_dir)

    zip_res(output_dir, output_file=f"{res_name}_submit.zip")

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")