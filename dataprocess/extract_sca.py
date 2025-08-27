"""
# Created: 2024-02-13 20:37
# Updated: 2024-04-17 21:08
# Copyright (C) 2024-now, Scania Sverige EEARP Group
# Author: Qingwen ZHANG  (https://kin-zhang.github.io/), Ajinkya Khoche (ajinkya.khoche@scania.com)
# License: GPLv2, allow it free only for academic use.

# Description: Preprocess Data, save as h5df format for faster loading

# Running Example: python extract_sca.py --origin_data data/HereProject --metadata_pkl data/scania_pseudo_infos.pkl --output_dir data/Scania/preprocess/train --nproc 16
"""

import os, sys, glob
from tqdm import tqdm
from pathlib import Path
import numpy as np
import fire, time

import h5py, json, yaml, pickle
from mmcv.ops import points_in_boxes_part
import torch

import multiprocessing
# ref: https://stackoverflow.com/questions/48822463/how-to-use-pytorch-multiprocessing
from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../OpenSceneFlow'))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index, cal_pose0to1Numpy
from src.utils.av2_eval import CATEGORY_TO_INDEX, NameMapping, BOUNDING_BOX_EXPANSION

def check_data(pts_filename):
    lack_name = None
    for attr in ['X', 'Y', 'Z', 'W', 'sensor', 'deltaT']:
        attr_path = pts_filename + '_' + attr + '.bin'
        if not os.path.isfile(attr_path):
            lack_name = f"{attr_path}"
            break
    return lack_name

def get_pc(pts_filename):
    lidar_attr = []
    lidar_id, lidar_dt = None, None
    for attr in ['X', 'Y', 'Z', 'W', 'sensor', 'deltaT']:
        attr_path = pts_filename + '_' + attr + '.bin'
        with open(attr_path, 'rb') as fid:
            # read deltaT and convert to seconds
            if attr == 'sensor':
                attr_array = np.fromfile(fid, np.int8)
                lidar_id = attr_array
            elif attr == 'deltaT':
                attr_array = np.fromfile(fid, np.int32) * 1e-9
                lidar_dt = attr_array
            else:
                attr_array = np.fromfile(fid, np.float32)
                lidar_attr.append(attr_array)
    return lidar_attr, lidar_id, lidar_dt

def get_pose_and_timestamp(sequence_meta, frame_idx):
    transform_matrix = np.eye(4)
    timestamp = int(sequence_meta['superframes'][frame_idx]['timestamp_epoch_ns'])
    smooth_pose = sequence_meta['superframes'][frame_idx]['smoothPosition']
    yaw = float(smooth_pose['smothYaw_rad'])
    transform_matrix[:3,:3] = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                        [np.sin(yaw), np.cos(yaw), 0],
                                        [0, 0, 1]])
    transform_matrix[0, 3] = float(smooth_pose['smoothX_m'])
    transform_matrix[1, 3] = float(smooth_pose['smoothY_m'])
    return transform_matrix, timestamp

def process_one(origin_data, output_dir: Path, scene_id, scene_meta):
    def create_group_data(group, pc, pc_id, pc_dt, SensorsCenter, pose, timestamp, anno_bbx=None, \
                          flow_0to1=None, flow_valid=None, flow_category=None, flow_instance_id=None, ego_motion=None):
        group.create_dataset('lidar', data=pc.astype(np.float32)) # x,y,z,intensity
        group.create_dataset('lidar_id', data=pc_id.astype(np.uint8)) # sensor id
        group.create_dataset('lidar_dt', data=pc_dt.astype(np.float32)) # deltaT
        group.create_dataset('SensorsCenter', data=SensorsCenter.astype(np.float32)) # lidar to superframe extrinsic, shape: [LiDAR_num, 3] (x, y, z)
        group.create_dataset('pose', data=pose.astype(np.float64))
        group.create_dataset('timestamp', data=timestamp)
        if flow_0to1 is not None:
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
        if flow_instance_id is not None:
            group.create_dataset('flow_instance_id', data=flow_instance_id.astype(np.uint32))
        if ego_motion is not None:
            group.create_dataset('ego_motion', data=ego_motion)
        if anno_bbx is not None:
            group.create_dataset('anno_bbx', data=anno_bbx.astype(np.float32))

    def compute_flow(pc0, pose0, pose1, annos, lidar_dt, lidar_dt1):
        ego1_SE3_ego0 = cal_pose0to1Numpy(pose0, pose1)
        flow = pc0[:,:3] @ ego1_SE3_ego0[:3,:3].T + ego1_SE3_ego0[:3,-1] - pc0[:,:3] # pose flow

        valid = np.ones(len(pc0), dtype=np.bool_)
        classes = np.zeros(len(pc0), dtype=np.uint8)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dimensions is size: l, w, h
        anno_box = torch.tensor(np.concatenate([annos['location'], annos['dimensions'], annos['heading'].reshape(-1,1)], axis=1), device=device)
        anno_box[:,2] -= anno_box[:,5]/2 # move center to ground
        anno_speed = torch.tensor(annos['speed'], device=device) # speed: Nx1
        anno_vel = torch.tensor(annos['velocity'], device=device) # velocity: Nx2
        anno_dt = annos['mean_delta_t']

        # FIXME(hardcode 0.1): enlarge along heading by vel * 0.1 sec (worst case delta t) * 2 (to cover front/back) + epsilon (margin of error)
        anno_box[anno_speed!=torch.inf,3] += anno_speed[anno_speed!=torch.inf]*0.1*2 + BOUNDING_BOX_EXPANSION
        # NOTE: expand width and height to accommodate all points
        anno_box[:,4] += 0.4
        anno_box[:,5] += BOUNDING_BOX_EXPANSION
        
        pc0_tensor = torch.tensor(pc0, device=device)
        instance_id0 = points_in_boxes_part(pc0_tensor[None,:,:3].double(), anno_box.double().unsqueeze(0)).squeeze()
        points_in_box_mask0 = (instance_id0!=-1).cpu().numpy()
        
        anno_vel_np = anno_vel[instance_id0[points_in_box_mask0]].cpu().numpy()
        anno_vel_3d = np.hstack([anno_vel_np, np.zeros(len(anno_vel_np)).reshape(-1,1)])
        anno_vel_per_pt = anno_vel_3d

        # NOTE: sometimes velocity is inf because only one box of an instance was observed. the flow for those points should be invalid
        valid[points_in_box_mask0] &= ~np.isinf(anno_vel_per_pt).any(axis=1)
        anno_vel_per_pt[np.isinf(anno_vel_per_pt).any(axis=1), :] = 0.

        instance_id0 = instance_id0.cpu().numpy().astype(np.int32)

        # TODO: this is an approximation. the actual flow should be velocity * (0.1 + dt1 - dt0),
        # where dt1 is mean lidar_dt of instance in pc1, and resp for dt0. also this needs to be done sensor-wise
        # Rough scene flow = vel0_per_pt * 0.1 (sweep interval)
        obj_flow = anno_vel_per_pt * 0.1
        flow[points_in_box_mask0,:] += obj_flow[:,:].astype(np.float32)

        # assign class ids
        annos['name'].append('none') # set: -1 index to no label
        name_array = np.array(annos['name'])[instance_id0]
        classes = np.array([CATEGORY_TO_INDEX[NameMapping[name_array[i]]] for i in range(len(name_array))])
        instance_id0 += 1 # change instance id background from -1 to 0, since points_in_boxes_part return background as -1
        
        return {'flow_0_1': flow,
                'valid_0': valid, 'classes_0': classes, 
                'instance_0': instance_id0.astype(np.uint32),
                'ego_motion': ego1_SE3_ego0}

    batch_frames = []
    for i, frame_folder in enumerate(sorted(os.listdir(os.path.join(origin_data, scene_id)))):
        # then it's not a frame folder
        if not frame_folder.startswith('superframe_'):
            continue
        batch_frames.append(frame_folder)
    batch_frames.sort()

    with h5py.File(output_dir/f'{scene_id}.h5', 'a') as f:

        if len(f.keys()) == len(batch_frames):
            print(f"{scene_id} already exist, skip. and the total timestamp is correct.")
            return
        
        meta_json_file = os.path.join(origin_data, scene_id) + f'/sequence_{int(scene_id.split("_")[1])}.json'
        if not os.path.exists(meta_json_file):
            print(f"{scene_id} has no meta file, skip.")
            return
        
        sequence_meta = json.load(open(meta_json_file))
        extrinsic_file = os.path.join(BASE_DIR, f"assets/private/lidar_ext/{sequence_meta['vehicle'].lower()}-generated.yml")
        data = yaml.safe_load(open(extrinsic_file))
        lidar_extrinsic_dict = {}
        for id_num in range(10): # 0-9, since mumble have 10 lidars, other three are 6 lidars.
            array_name = f'lidarArray_arrayEl{id_num}'
            if array_name not in data['parameters']:
                continue
            sensor_center = data['parameters'][array_name]['nominalPosition']
            lidar_extrinsic_dict[data['parameters'][array_name]['humanReadableReference']] = [sensor_center['x'], sensor_center['y'], sensor_center['z']]

        for i, one_frame in enumerate(batch_frames):
            pts_filename = os.path.join(origin_data, scene_id, one_frame, f'{one_frame}')
            lack_name = check_data(pts_filename)
            if lack_name is not None:
                print(f"{scene_id} has no data file: {lack_name}")
                break

            frame_idx = int(one_frame.split('_')[-1]) - 1  # JSON is 0-based index, while one_frame name (superframe_*) is 1-based index
            
            # Step 1: the lidar data
            lidar_attr, lidar_id, lidar_dt = get_pc(pts_filename)
            pc = np.array(lidar_attr).T # shape: [N, 4]

            # Step 2: sensor center
            SensorsCenter = []
            lidar_id_list = np.unique(lidar_id)
            for lidar_single_id in lidar_id_list:
                SensorsCenter.append(lidar_extrinsic_dict[sequence_meta['lidars'][f'lidar{lidar_single_id-1}']['name']])
            
            # Step 3: pose data
            pose, timestamp = get_pose_and_timestamp(sequence_meta=sequence_meta, frame_idx=frame_idx)

            # Step4: next frame
            if i >= len(scene_meta) - 1:
                group = f.create_group(one_frame.split('_')[-1])
                # group, pc, pc_id, pc_dt, SensorsCenter, pose, timestamp
                create_group_data(group=group, pc=pc, pc_id=lidar_id, pc_dt=lidar_dt, \
                                    SensorsCenter=np.array(SensorsCenter), pose=pose, timestamp=timestamp)
                continue

            # else:
            annos = scene_meta[i]['annos']
            next_frame = batch_frames[i+1]
            pts_filename = os.path.join(origin_data, scene_id, next_frame, f'{next_frame}')
            lack_name = check_data(pts_filename)
            if lack_name is not None:
                print(f"{scene_id} has no data file: {lack_name}")
                break
            frame_idx = int(next_frame.split('_')[-1]) - 1  # JSON is 0-based index

            # read the lidar data
            lidar_attr, lidar_id1, lidar_dt1 = get_pc(pts_filename=pts_filename)
            # pc1 = np.array(lidar_attr).T # shape: [N, 4]

            # read the pose data
            pose1, timestamp1 = get_pose_and_timestamp(sequence_meta=sequence_meta, frame_idx=frame_idx)
            # annos1 = scene_meta[i+1]['annos']

            # compute sceneflow
            scene_flow = compute_flow(pc, pose, pose1, annos, lidar_dt, lidar_dt1)
            
            # save the data
            group = f.create_group(one_frame.split('_')[-1])
            create_group_data(group=group, pc=pc, pc_id=lidar_id, pc_dt=lidar_dt, \
                              SensorsCenter=np.array(SensorsCenter), \
                              pose=pose.astype(np.float32), timestamp=timestamp, \
                              flow_0to1=scene_flow['flow_0_1'], flow_valid=scene_flow['valid_0'], flow_category=scene_flow['classes_0'], flow_instance_id=scene_flow['instance_0'], \
                              ego_motion=scene_flow['ego_motion'].astype(np.float32)
                            )

def proc(x):
    process_one(*x)

def main(
    origin_data: str = "/home/kin/data/Scania/val",
    metadata_pkl: str = "/home/kin/data/Scania/scania_pseudo_infos.pkl",
    output_dir: str ="/home/kin/data/Scania/preprocess/val_debuging",
    nproc: int = (multiprocessing.cpu_count() - 1),
    create_index_only: bool = False,
):
    if create_index_only:
        create_reading_index(Path(output_dir))
        return

    with open(metadata_pkl, 'rb') as f:
        metadata = pickle.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_scenes = []
    all_metadata = []
    for scene_id in sorted(os.listdir(origin_data)):
        # if the data is not folder, skip
        if os.path.isdir(os.path.join(origin_data, scene_id)) is not True or 'batch' not in scene_id:
            continue
        # # if the h5df is already exist and the total timestamp is correct, then skip
        # TODO: add the check for h5df
        meta = [l for l in metadata if l['sample_idx']==scene_id]
        if len(meta) != 0:
            all_metadata.append(meta)
            all_scenes.append(scene_id)

    all_scenes.sort()
    args = [(origin_data, Path(output_dir), all_scenes[i], all_metadata[i]) for i in range(len(all_scenes))]
    print(f'Using {nproc} processes for creating {len(all_scenes)} scene.')

    # # for debug
    # for x in tqdm(args):
    #     process_one(*x)
    #     break

    if nproc <= 1:
        for x in tqdm(args):
            process_one(*x)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(all_scenes), ncols=100))
    
    create_reading_index(Path(output_dir))

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nRunning {__file__} used: {(time.time() - start_time)/60:.2f} mins")