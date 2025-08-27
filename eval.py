"""
# Created: 2024-04-15 17:32
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)

# Description: print score based on the data file, need save result first. 
#              Run 4_vis.py to produce output append in existing .h5 file.
"""

import numpy as np
import fire, time, json
from tqdm import tqdm
from tabulate import tabulate

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../OpenSceneFlow' ))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, egopts_mask
from src.utils.av2_eval import CLOSE_DISTANCE_THRESHOLD, BUCKETED_METACATAGORIES, CATEGORY_TO_INDEX

class InstanceMetrics:
    def __init__(self, data_name, min_vel = 1.5, sensor_hz = 10.0):
        self.class_names = self.init_class_data()
        self.min_vel = min_vel # since inside min_flow (default: 15cm) may not cause error label?
        self.frame_cnt = 0
        self.sensor_dt = 1.0 / sensor_hz
        self.data_name = data_name
    
    def init_class_data(self):
        init_data = lambda: {'num_pts': [], 'mpe': [], 'cham': [], 'std_mpe': [], 'std_cham': []}
        dict_data = {"CAR": {}, "OTHER_VEHICLES": {}}
        for cats_name in ["CAR", "OTHER_VEHICLES"]: # , "PEDESTRIAN", "WHEELED_VRU"
            dict_data[cats_name]['vel'] = {range_name: init_data() for range_name in ['0-10', '10-20', '20-30', '30+']}
            dict_data[cats_name]['dis'] = {range_name: init_data() for range_name in ['0-10', '10-20', '20-30', '30+']}
            dict_data[cats_name]['mean'] = init_data()
        return dict_data

    def flow2compDis(self, flow, dt0):
        """
        refine points belong to a instance based on the flow and dt0.
        dt0: nanosecond offsets _from_ the start of the sweep. 
            * The first point dt is normally 0.0 since it's the start of the sweep.
            * So the latest point dt is normally T_sensor like 0.1.
        """
        return flow/self.sensor_dt * dt0[:, None] 

    def refine_pts(self, pc, ds):
        ref_pc = pc[:,:3] + ds
        return ref_pc
        
    def cal_cham(self, pc1, pc2):
        """
        definition of chamfer distance: https://github.com/UM-ARM-Lab/Chamfer-Distance-API?tab=readme-ov-file#chamfer-distance-api
        """
        distance_matrix_12 = np.linalg.norm(pc1[:, None] - pc2, axis=2)
        distance_matrix_21 = np.linalg.norm(pc2[:, None] - pc1, axis=2)
        cham = (np.nanmean(np.min(distance_matrix_12, axis=1)) + np.nanmean(np.min(distance_matrix_21, axis=1))) / 2
        return cham

    def step(self, pc, est_flow, gt_flow, pc_dt0, gt_category, gt_instance):

        frame_score = self.init_class_data()
        refine_pc = self.refine_pts(pc, self.flow2compDis(est_flow, pc_dt0))
        # NOTE: if you can directly output distance please do:
        # refine_pc = self.refine_pts(pc, est_ds)
        
        gt_refine_pc = self.refine_pts(pc, self.flow2compDis(gt_flow, pc_dt0))

        # different categories
        for cats_name in ["CAR", "OTHER_VEHICLES"]:
            classes_ids = [CATEGORY_TO_INDEX[cat] for cat in BUCKETED_METACATAGORIES[cats_name]]
            mask_class = np.isin(gt_category, np.array(classes_ids))
            class_pts_num = np.sum(mask_class)
            if class_pts_num == 0:
                continue
            
            mask_gt_ins = gt_instance[mask_class]
            gt_flow_class = gt_flow[mask_class]
            refine_pc_class = refine_pc[mask_class]
            gt_refine_pc_class = gt_refine_pc[mask_class]
            
            # different instances, cluster_i
            for instance_id in np.unique(mask_gt_ins):
                mask = (mask_gt_ins == instance_id)
                num_pts = np.sum(mask)
                vel_ins = np.linalg.norm(gt_flow_class[mask], axis=1).mean() / self.sensor_dt
                if num_pts < 10 or vel_ins < self.min_vel: # skip cluster_i with less than 10 points
                    continue
                dis_ins = np.linalg.norm(pc[mask_class][mask], axis=1).mean()
                mpe = np.linalg.norm(gt_refine_pc_class[mask] - refine_pc_class[mask], axis=1).mean()
                cham = self.cal_cham(gt_refine_pc_class[mask], refine_pc_class[mask])

                # Categorize based on velocity and distance
                for metric, values in [('vel', vel_ins), ('dis', dis_ins)]:
                    range_name = None
                    if 0 < values < 10:
                        range_name = '0-10'
                    elif 10 <= values < 20:
                        range_name = '10-20'
                    elif 20 <= values < 30:
                        range_name = '20-30'
                    elif values >= 30:
                        range_name = '30+'
                    if range_name is None:
                        print('--- [ERROR]: range_name is None --- the value is:', values, ' in ', metric)
                        continue
                    frame_score[cats_name][metric][range_name]['num_pts'].append(num_pts)
                    frame_score[cats_name][metric][range_name]['mpe'].append(mpe)
                    frame_score[cats_name][metric][range_name]['cham'].append(cham)
        
        for cats_name in frame_score:
            total_num_list, mpe_list, cham_list = [], [], []
            for metric in ['vel', 'dis']:
                for range_name in frame_score[cats_name][metric]:
                    num_pts_list = frame_score[cats_name][metric][range_name]['num_pts']
                    if len(num_pts_list) > 0:
                        total_num_pts = sum(num_pts_list)
                        avg_mpe = np.average(frame_score[cats_name][metric][range_name]['mpe'], weights=num_pts_list)
                        avg_cham = np.average(frame_score[cats_name][metric][range_name]['cham'], weights=num_pts_list)
                        self.class_names[cats_name][metric][range_name]['num_pts'] += num_pts_list
                        self.class_names[cats_name][metric][range_name]['mpe'] += frame_score[cats_name][metric][range_name]['mpe']
                        self.class_names[cats_name][metric][range_name]['cham'] += frame_score[cats_name][metric][range_name]['cham']
                        
                        if metric == 'vel': # only add once.
                            mpe_list.append(avg_mpe)
                            cham_list.append(avg_cham)
                            total_num_list.append(total_num_pts)

            # Update mean values from vel
            num_pts = sum(total_num_list)
            if num_pts == 0:
                continue
            mean_mpe = np.nanmean(mpe_list)
            mean_cham = np.nanmean(cham_list)
            std_mpe = np.nanstd(mpe_list)
            std_cham = np.nanstd(cham_list)

            self.class_names[cats_name]['mean']['num_pts'].append(num_pts)
            self.class_names[cats_name]['mean']['mpe'].append(mean_mpe)
            self.class_names[cats_name]['mean']['cham'].append(mean_cham)
            self.class_names[cats_name]['mean']['std_mpe'].append(std_mpe)
            self.class_names[cats_name]['mean']['std_cham'].append(std_cham)

        self.frame_cnt += 1
    
    def print(self, flow_mode="flow", file_name="result_av2.json"):
        """
        print the statistics of the refinement metrics. And save the results to a json file.
        """
        def savejson(overall_data, vel_data, dis_data, flow_mode, category_name):
            if os.path.exists(file_name):
                with open(file_name, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            if self.data_name not in data:
                data[self.data_name] = {}
            if flow_mode not in data[self.data_name]:
                data[self.data_name][flow_mode] = {}

            data[self.data_name][flow_mode][category_name] = {
                    "overall": {"mpe": overall_data[0][1], "cd": overall_data[0][2], "std_mpe": overall_data[0][3], "std_cd": overall_data[0][4], "num_pts": int(overall_data[0][5]), "num_obj": int(overall_data[0][6])},
                    "velocity": {"0-10": {"mpe": vel_data[0][1], "cd": vel_data[0][2], "num_pts": int(vel_data[0][3]), "num_obj": int(vel_data[0][4])},
                                "10-20": {"mpe": vel_data[1][1], "cd": vel_data[1][2], "num_pts": int(vel_data[1][3]), "num_obj": int(vel_data[1][4])},
                                "20-30": {"mpe": vel_data[2][1], "cd": vel_data[2][2], "num_pts": int(vel_data[2][3]), "num_obj": int(vel_data[2][4])},
                                "30+": {"mpe": vel_data[3][1], "cd": vel_data[3][2], "num_pts": int(vel_data[3][3]), "num_obj": int(vel_data[3][4])}
                                },
                    "distance": {"0-10": {"mpe": dis_data[0][1], "cd": dis_data[0][2], "num_pts": int(dis_data[0][3]), "num_obj": int(dis_data[0][4])},
                                "10-20": {"mpe": dis_data[1][1], "cd": dis_data[1][2], "num_pts": int(dis_data[1][3]), "num_obj": int(dis_data[1][4])},
                                "20-30": {"mpe": dis_data[2][1], "cd": dis_data[2][2], "num_pts": int(dis_data[2][3]), "num_obj": int(dis_data[2][4])},
                                "30+": {"mpe": dis_data[3][1], "cd": dis_data[3][2], "num_pts": int(dis_data[3][3]), "num_obj": int(dis_data[3][4])}
                                }
            }
            with open(file_name, "w") as f:
                json.dump(data, f, indent=4)
        def safe_average(values, weights):
            return np.average(values, weights=weights) if len(values) > 0 else "N/A"
        def safe_std(values):
            return np.std(values) if len(values) > 0 else "N/A"
        
        def print_category_data(category_data, category_name):
            print(f"{category_name} Statistics:")
            overall_data = [
                [category_name, 
                safe_average(category_data['mean']['mpe'], category_data['mean']['num_pts']),
                safe_average(category_data['mean']['cham'], category_data['mean']['num_pts']),
                safe_std(category_data['mean']['std_mpe']),
                safe_std(category_data['mean']['std_cham']),
                np.sum(category_data['mean']['num_pts']),
                # instance count
                len(category_data['mean']['num_pts'])
                ]
            ]
            print(tabulate(overall_data, headers=["Class", "MPE", "CD", "Std MPE", "Std CD", "NumPts", "#ins"], tablefmt="pretty"))

            print("Velocity Ranges:")
            vel_data = []
            for range_name in ['0-10', '10-20', '20-30', '30+']:
                mpe_values = category_data['vel'][range_name]['mpe']
                cham_values = category_data['vel'][range_name]['cham']
                num_pts = category_data['vel'][range_name]['num_pts']
                vel_data.append([
                    range_name,
                    safe_average(mpe_values, num_pts),
                    safe_average(cham_values, num_pts),
                    np.sum(num_pts),
                    len(num_pts)
                ])
            print(tabulate(vel_data, headers=["Range", "MPE", "CD", "NumPts", "#ins"], tablefmt="pretty"))

            print("Distance Ranges:")
            dis_data = []
            for range_name in ['0-10', '10-20', '20-30', '30+']:
                mpe_values = category_data['dis'][range_name]['mpe']
                cham_values = category_data['dis'][range_name]['cham']
                num_pts = category_data['dis'][range_name]['num_pts']
                dis_data.append([
                    range_name,
                    safe_average(mpe_values, num_pts),
                    safe_average(cham_values, num_pts),
                    np.sum(num_pts),
                    len(num_pts)
                ])
            print(tabulate(dis_data, headers=["Range", "MPE", "CD", "NumPts", "#ins"], tablefmt="pretty"))
            
            savejson(overall_data, vel_data, dis_data, flow_mode, category_name)
            
        print(f"HiMo refinement metrics for {flow_mode} method in {self.data_name} data:")
        
        for cats_name in ["CAR", "OTHER_VEHICLES"]: # , "PEDESTRIAN", "WHEELED_VRU"
            if len(self.class_names[cats_name]['mean']['num_pts']) > 0:
                print_category_data(self.class_names[cats_name], cats_name)

        print(f"Total frames processed: {self.frame_cnt} for {flow_mode} method in {self.data_name} data.")
    

def main(
    # data_dir: str ="/home/kin/data/Scania/preprocess/val",
    data_dir: str ="/home/kin/data/av2/h5py/sensor/himo_val",
    flow_mode: str = "raw", 
):
    # FIXME: maybe users' path is different.... change this!!!
    data_name = data_dir.split("/")[4].lower()

    # NOTE: 1.5 for scania, 3.0 for av2
    refinement_metrics = InstanceMetrics(min_vel = 3.0, data_name=data_name)
    dataset = HDF5Data(data_dir, res_name=flow_mode, eval=True)
    for data_id in tqdm(range(0, len(dataset)), ncols=80, desc=f"Evaluating {flow_mode} on {data_name}"):
        data = dataset[data_id]
        pc0 = data['pc0']
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        
        est_flow = np.zeros_like(pose_flow) if flow_mode == "raw" else (data[flow_mode] - pose_flow)
        gt_flow = data['flow'] - pose_flow

        pc_dis = np.linalg.norm(pc0[:, :2], axis=1)
        dis_mask = pc_dis <= CLOSE_DISTANCE_THRESHOLD
        notgm_mask = ~data['ground_mask0']
        # scania
        if data_name == "scania":
            mask_eval = dis_mask & data['flow_is_valid'] & notgm_mask & egopts_mask(pc0)
        else:
            mask_eval = dis_mask & notgm_mask & egopts_mask(pc0, min_bound=[-1.5, -1.5, -2.0], max_bound=[1.5, 1.5, 2.0])

        dt0 = max(data['dt0']) - data['dt0'] # we want to see the last frame. dts: (N,1) Nanosecond offsets _from_ the start of the sweep.
        refinement_metrics.step(pc0[mask_eval,:], est_flow[mask_eval,:], gt_flow[mask_eval,:], \
                                dt0[mask_eval], data['flow_category_indices'][mask_eval], data['flow_instance_id'][mask_eval])
        
    refinement_metrics.print(flow_mode=flow_mode, file_name=f"result_{data_name}.json")

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")