"""
# Created: 2024-04-15 17:32
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)

# Description: print compensation score based on flow result or 
# provided zip file, both need save result first. 
#
"""

import numpy as np
import fire, time, json
from tqdm import tqdm
from tabulate import tabulate
from scipy.spatial import cKDTree
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), 'OpenSceneFlow' ))
sys.path.append(BASE_DIR)

from src.dataset import HDF5Dataset
from src.utils.av2_eval import CLOSE_DISTANCE_THRESHOLD, BUCKETED_METACATAGORIES, CATEGORY_TO_INDEX
from utils import flow2compDis, refine_pts, ego_pts_mask, check_valid
from save_zip import read_output_zip
class InstanceMetrics:
    def __init__(self, data_name, sensor_hz = 10.0):
        self.frame_cnt = 0
        self.sensor_dt = 1.0 / sensor_hz
        self.data_name = data_name

        # NOTE for custom min_vel: 
        # 1. scania data have some error-label for small moving (default: 15cm)
        # 2. normal dataset (1-2 lidar), small flow evaluation didn't have distortion.
        if data_name in ["scania"]:
            self.min_vel = 1.5
        else:
            self.min_vel = 3.0

        self.evaluate_data = self.init_evaluate_data()

    def init_evaluate_data(self):
        """Initialize the data structure to hold evaluation metrics."""
        init_data = lambda: {'num_pts': [], 'mpe': [], 'cham': [], 'std_mpe': [], 'std_cham': []}
        dict_data = {"CAR": {}, "OTHER_VEHICLES": {}}
        for cats_name in ["CAR", "OTHER_VEHICLES"]: # , "PEDESTRIAN", "WHEELED_VRU"
            dict_data[cats_name]['vel'] = {range_name: init_data() for range_name in ['0-10', '10-20', '20-30', '30+']}
            dict_data[cats_name]['dis'] = {range_name: init_data() for range_name in ['0-10', '10-20', '20-30', '30+']}
            dict_data[cats_name]['mean'] = init_data()
        return dict_data

    def cal_chamfer(self, pc1: np.ndarray, pc2: np.ndarray) -> float:
        """Calculate Chamfer distance using KD-trees (memory-efficient).

        Chamfer distance = (mean(min_dist(pc1->pc2)) + mean(min_dist(pc2->pc1))) / 2
        """
        if len(pc1) == 0 or len(pc2) == 0:
            return float('nan')
        tree2 = cKDTree(pc2)
        d12, _ = tree2.query(pc1, k=1)
        tree1 = cKDTree(pc1)
        d21, _ = tree1.query(pc2, k=1)
        cham = (np.nanmean(d12) + np.nanmean(d21)) / 2.0
        return float(cham)

    def step_eval(self, pc, gt_flow, pc_dt0, gt_category, gt_instance, est_flow=None, est_dis=None):

        frame_score = self.init_evaluate_data()
        if est_flow is not None:
            refine_pc = refine_pts(pc, flow2compDis(est_flow, pc_dt0, sensor_dt=self.sensor_dt))
        elif est_dis is not None:
            refine_pc = refine_pts(pc, est_dis)

        gt_refine_pc = refine_pts(pc, flow2compDis(gt_flow, pc_dt0, sensor_dt=self.sensor_dt))

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
                cham = self.cal_chamfer(gt_refine_pc_class[mask], refine_pc_class[mask])

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
                        self.evaluate_data[cats_name][metric][range_name]['num_pts'] += num_pts_list
                        self.evaluate_data[cats_name][metric][range_name]['mpe'] += frame_score[cats_name][metric][range_name]['mpe']
                        self.evaluate_data[cats_name][metric][range_name]['cham'] += frame_score[cats_name][metric][range_name]['cham']
                        
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

            self.evaluate_data[cats_name]['mean']['num_pts'].append(num_pts)
            self.evaluate_data[cats_name]['mean']['mpe'].append(mean_mpe)
            self.evaluate_data[cats_name]['mean']['cham'].append(mean_cham)
            self.evaluate_data[cats_name]['mean']['std_mpe'].append(std_mpe)
            self.evaluate_data[cats_name]['mean']['std_cham'].append(std_cham)

        self.frame_cnt += 1
    
    def print(self, flow_mode="flow", file_name="result_av2.json"):
        # --- Helper: Save detailed metrics to JSON (Preserves original structure) ---
        def savejson(overall_data, vel_data, dis_data, flow_mode, category_name):
            if os.path.exists(file_name):
                with open(file_name, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}
            else:
                data = {}

            if self.data_name not in data:
                data[self.data_name] = {}
            if flow_mode not in data[self.data_name]:
                data[self.data_name][flow_mode] = {}

            # Construct payload matching original format
            entry = {
                "overall": {
                    "mpe": overall_data[0][1], "cd": overall_data[0][2],
                    "std_mpe": overall_data[0][3], "std_cd": overall_data[0][4],
                    "num_pts": int(overall_data[0][5]), "num_obj": int(overall_data[0][6])
                },
                "velocity": {},
                "distance": {}
            }
            
            # Map list data to dictionary keys (0-10, 10-20, etc.)
            ranges = ["0-10", "10-20", "20-30", "30+"]
            for i, r_name in enumerate(ranges):
                entry["velocity"][r_name] = {
                    "mpe": vel_data[i][1], "cd": vel_data[i][2],
                    "num_pts": int(vel_data[i][3]), "num_obj": int(vel_data[i][4])
                }
                entry["distance"][r_name] = {
                    "mpe": dis_data[i][1], "cd": dis_data[i][2],
                    "num_pts": int(dis_data[i][3]), "num_obj": int(dis_data[i][4])
                }
            
            data[self.data_name][flow_mode][category_name] = entry
            with open(file_name, "w") as f:
                json.dump(data, f, indent=4)
        def safe_average(values, weights):
            return np.average(values, weights=weights) if len(values) > 0 and np.sum(weights) > 0 else 0.0
        def safe_std(values):
            return np.std(values) if len(values) > 0 else 0.0

        # --- Main Logic ---
        target_cats = ["CAR", "OTHER_VEHICLES"]
        cat_display_map = {"CAR": "CAR", "OTHER_VEHICLES": "OTHERS"}
        ranges = ['0-10', '10-20', '20-30', '30+']
        
        # Container to aggregate Total stats
        total_data = {"mpe": [], "cham": [], "std_mpe": [], "std_cham": [], "num_pts": []}
        table_rows = []

        print(f"\nHiMo refinement metrics for {flow_mode} in {self.data_name}:")

        for cat in target_cats:
            if cat not in self.evaluate_data or len(self.evaluate_data[cat]['mean']['num_pts']) == 0:
                continue
            
            raw = self.evaluate_data[cat]
            mean_raw = raw['mean']

            # 1. Calc Overall Stats for current category
            cat_mpe = safe_average(mean_raw['mpe'], mean_raw['num_pts'])
            cat_cd = safe_average(mean_raw['cham'], mean_raw['num_pts'])
            cat_std_mpe = safe_std(mean_raw['std_mpe'])
            cat_std_cd = safe_std(mean_raw['std_cham'])
            cat_n_pts = np.sum(mean_raw['num_pts'])
            cat_n_obj = len(mean_raw['num_pts'])

            # 2. Accumulate to Total
            total_data['mpe'].extend(mean_raw['mpe'])
            total_data['cham'].extend(mean_raw['cham'])
            total_data['num_pts'].extend(mean_raw['num_pts'])
            # We don't necessarily need std list for Total unless calculating pooled std, but keeping for consistency
            
            # 3. Prepare data for SaveJson (Detailed breakdowns)
            overall_entry = [[cat, cat_mpe, cat_cd, cat_std_mpe, cat_std_cd, cat_n_pts, cat_n_obj]]
            vel_entries, dis_entries = [], []
            
            for r in ranges:
                v = raw['vel'][r]
                vel_entries.append([r, safe_average(v['mpe'], v['num_pts']), safe_average(v['cham'], v['num_pts']), np.sum(v['num_pts']), len(v['num_pts'])])
                d = raw['dis'][r]
                dis_entries.append([r, safe_average(d['mpe'], d['num_pts']), safe_average(d['cham'], d['num_pts']), np.sum(d['num_pts']), len(d['num_pts'])])

            savejson(overall_entry, vel_entries, dis_entries, flow_mode, cat)

            table_rows.append([
                cat_display_map.get(cat, cat),
                f"{cat_cd:.3f} ± {cat_std_cd:.2f}",
                f"{cat_mpe:.3f} ± {cat_std_mpe:.2f}",
                int(cat_n_pts),
                int(cat_n_obj)
            ])

        if len(total_data['num_pts']) > 0:
            tot_mpe = safe_average(total_data['mpe'], total_data['num_pts'])
            tot_cd = safe_average(total_data['cham'], total_data['num_pts'])
            
            total_row = [
                "Total",
                f"{tot_cd:.3f}",   # Total usually displayed without std in paper tables
                f"{tot_mpe:.3f}",
                int(np.sum(total_data['num_pts'])),
                int(len(total_data['num_pts']))
            ]
            table_rows.insert(0, total_row)

        # --- Print Final Table ---
        headers = ["Class", "CDE (Chamfer) ↓", "MPE (Point Err) ↓", "# Points", "# Objs"]
        print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid", stralign="center"))
        print(f"Total frames processed: {self.frame_cnt}")
        print(f"Results saved to {file_name}\n")

def main(
    # data_dir: str = "/home/kin/data/Scania/preprocess/val",
    data_dir: str = "/home/kin/data/av2/h5py/sensor/himo",
    flow_mode: str = "",
    comp_dis_zip: str = "",
):
    data_name, EVAL_FLAG = check_valid(data_dir, flow_mode, comp_dis_zip)

    refinement_metrics = InstanceMetrics(data_name=data_name)
    dataset = HDF5Dataset(data_dir, vis_name=flow_mode if EVAL_FLAG == 2 else '', eval=True)

    for data_id in tqdm(range(0, len(dataset)), ncols=80, desc=f"Evaluating {flow_mode} on {data_name}"):
        data = dataset[data_id]
        pc0, pose0, pose1 = data['pc0'], data['pose0'], data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        gt_flow = data['flow'] - pose_flow

        pc_dis = np.linalg.norm(pc0[:, :2], axis=1)
        dis_mask = pc_dis <= CLOSE_DISTANCE_THRESHOLD
        notgm_mask = ~data['gm0']

        # scania
        if data_name == "scania":
            mask_eval = dis_mask & data['flow_is_valid'] & notgm_mask & ego_pts_mask(pc0)
        else:
            mask_eval = dis_mask & notgm_mask & ego_pts_mask(pc0, min_bound=[-1.5, -1.5, -2.0], max_bound=[1.5, 1.5, 2.0])
        
        # NOTE: we compensated to the latest observation. dts: (N,1) Nanosecond offsets _from_ the start of the sweep.
        dt0 = max(data['lidar_dt']) - data['lidar_dt']
        
        if EVAL_FLAG == 2: 
            est_flow = np.zeros_like(pose_flow) if flow_mode == "raw" else (data[flow_mode] - pose_flow)
            refinement_metrics.step_eval(pc0[mask_eval,:], gt_flow[mask_eval,:], \
                                    dt0[mask_eval], data['flow_category_indices'][mask_eval], data['flow_instance_id'][mask_eval],
                                    est_flow=est_flow[mask_eval,:])
        elif EVAL_FLAG == 1:
            comp_dis = read_output_zip(comp_dis_zip, (data['scene_id'], str(data['timestamp'])))
            refinement_metrics.step_eval(pc0[mask_eval,:], gt_flow[mask_eval,:], \
                                    dt0[mask_eval], data['flow_category_indices'][mask_eval], data['flow_instance_id'][mask_eval],
                                    est_dis=comp_dis[mask_eval,:])

    refinement_metrics.print(flow_mode=flow_mode, file_name=f"res-{data_name}.json")

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")