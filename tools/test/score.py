"""
# Created: 2024-04-15 17:32
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)

# Description: Standalone scoring program for CodaBench evaluation.
#              Compares GT and predicted compensation distances from zip files.
"""

# Install pyarrow if not available (CodaBench environment)
import subprocess
import sys
try:
    import pyarrow
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "-q"])

import numpy as np
import json, os
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
from tabulate import tabulate

# Annotation categories for Argoverse 2 (ordered for index mapping)
ANNOTATION_CATEGORIES = [
    "ANIMAL",
    "ARTICULATED_BUS",
    "BICYCLE",
    "BICYCLIST",
    "BOLLARD",
    "BOX_TRUCK",
    "BUS",
    "CONSTRUCTION_BARREL",
    "CONSTRUCTION_CONE",
    "DOG",
    "LARGE_VEHICLE",
    "MESSAGE_BOARD_TRAILER",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "OFFICIAL_SIGNALER",
    "PEDESTRIAN",
    "RAILED_VEHICLE",
    "REGULAR_VEHICLE",
    "SCHOOL_BUS",
    "SIGN",
    "STOP_SIGN",
    "STROLLER",
    "TRAFFIC_LIGHT_TRAILER",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "WHEELCHAIR",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
]

# Build category to index mapping (NONE=0, then 1-indexed categories)
CATEGORY_TO_INDEX = {"NONE": 0}
CATEGORY_TO_INDEX.update({cat: i + 1 for i, cat in enumerate(ANNOTATION_CATEGORIES)})

PEDESTRIAN_CATEGORIES = ["PEDESTRIAN", "STROLLER", "WHEELCHAIR", "OFFICIAL_SIGNALER"]
WHEELED_VRU = [
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
]
CAR = ["REGULAR_VEHICLE"]
OTHER_VEHICLES = [
    "BOX_TRUCK",
    "LARGE_VEHICLE",
    "RAILED_VEHICLE",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "ARTICULATED_BUS",
    "BUS",
    "SCHOOL_BUS",
]
BACKGROUND_CATEGORIES = ["NONE"]
BUCKETED_METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "CAR": CAR,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "WHEELED_VRU": WHEELED_VRU,
    "OTHER_VEHICLES": OTHER_VEHICLES,
}

def read_data_file(data_path: str, sweep_uuid: tuple) -> tuple:
    """Read compensation distance from a zip file or extracted directory.
    
    Args:
        data_path: Path to the zip file or extracted directory.
        sweep_uuid: (scene_id, timestamp) tuple.
    
    Returns:
        comp_dis: (N, 3) compensation distance.
        eval_mask: (N,) boolean mask for evaluation.
        flow_category: (N,) category indices.
        flow_instance: (N,) instance IDs.
        gt_flow_norm: (N,) GT flow magnitude.
        pc0: (N, 3) original point cloud coordinates (if available).
    """
    feather_path = f"{sweep_uuid[0]}/{sweep_uuid[1]}.feather"
    data_path = Path(data_path)
    
    # Check if it's a directory (extracted) or zip file
    if data_path.is_dir():
        # Read from extracted directory
        full_path = data_path / feather_path
        df = pd.read_feather(full_path)
    else:
        # Read from zip file
        with ZipFile(data_path, 'r') as myzip:
            with myzip.open(feather_path) as f:
                df = pd.read_feather(BytesIO(f.read()))
    
    comp_dis = np.stack([
        df['comp_dis_x_m'].values.astype(np.float32),
        df['comp_dis_y_m'].values.astype(np.float32),
        df['comp_dis_z_m'].values.astype(np.float32),
    ], axis=1)
    
    eval_mask = df['eval_mask'].values.astype(bool) if 'eval_mask' in df.columns else np.ones(len(comp_dis), dtype=bool)
    flow_category = df['flow_category_indices'].values.astype(np.uint8) if 'flow_category_indices' in df.columns else None
    flow_instance = df['flow_instance_id'].values.astype(np.uint32) if 'flow_instance_id' in df.columns else None
    # optional: gt flow magnitude (if the GT saving script included it)
    gt_flow_norm = df['gt_flow_norm'].values.astype(np.float32) if 'gt_flow_norm' in df.columns else None
    # optional: original point cloud (for Chamfer calculation)
    pc0 = None
    if 'pc0_x' in df.columns and 'pc0_y' in df.columns and 'pc0_z' in df.columns:
        pc0 = np.stack([
            df['pc0_x'].values.astype(np.float32),
            df['pc0_y'].values.astype(np.float32),
            df['pc0_z'].values.astype(np.float32),
        ], axis=1)
    return comp_dis, eval_mask, flow_category, flow_instance, gt_flow_norm, pc0


def list_sweep_uuids(data_path: str) -> list:
    """List all sweep UUIDs in a zip file or extracted directory.
    
    Returns:
        List of (scene_id, timestamp) tuples.
    """
    sweep_uuids = []
    data_path = Path(data_path)
    
    # Check if it's a directory (extracted) or zip file
    if data_path.is_dir():
        # Read from extracted directory
        for feather_file in data_path.rglob('*.feather'):
            # Get relative path parts
            rel_path = feather_file.relative_to(data_path)
            parts = rel_path.parts
            if len(parts) == 2:
                scene_id = parts[0]
                timestamp = parts[1].replace('.feather', '')
                sweep_uuids.append((scene_id, timestamp))
    else:
        # Read from zip file
        with ZipFile(data_path, 'r') as myzip:
            for name in myzip.namelist():
                if name.endswith('.feather'):
                    parts = name.split('/')
                    if len(parts) == 2:
                        scene_id, timestamp_file = parts
                        timestamp = timestamp_file.replace('.feather', '')
                        sweep_uuids.append((scene_id, timestamp))
    return sweep_uuids


def cal_chamfer(pc1: np.ndarray, pc2: np.ndarray) -> float:
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


def cal_mpe(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """Calculate Mean Point Error (L2 distance) between aligned point clouds."""
    return np.linalg.norm(pc1 - pc2, axis=1).mean()


class ScoreMetrics:
    """Metrics class for CodaBench scoring.

    This implements a per-instance, per-category, per-velocity-range evaluation 
    matching `eval.py` exactly. Key aggregation logic:
    1. Group instances by velocity range (0-10, 10-20, 20-30, 30+ m/s)
    2. Compute weighted average MPE/CDE per range (weighted by num_pts)
    3. Take nanmean across velocity ranges (unweighted) for final score
    """
    
    def __init__(self):
        self.frame_cnt = 0  # number of frames (sweeps) processed
        self.evaluate_data = self._init_evaluate_data()
    
    def _init_evaluate_data(self):
        """Initialize the data structure to hold evaluation metrics (matching eval.py)."""
        init_data = lambda: {'num_pts': [], 'mpe': [], 'cham': []}
        dict_data = {"CAR": {}, "OTHER_VEHICLES": {}}
        for cats_name in ["CAR", "OTHER_VEHICLES"]:
            dict_data[cats_name]['vel'] = {range_name: init_data() for range_name in ['0-10', '10-20', '20-30', '30+']}
            dict_data[cats_name]['mean'] = {'num_pts': [], 'mpe': [], 'cham': [], 'std_mpe': [], 'std_cham': []}
        return dict_data
    
    def step(self, gt_dis: np.ndarray, est_dis: np.ndarray, eval_mask: np.ndarray,
             gt_category: np.ndarray = None, gt_instance: np.ndarray = None,
             gt_flow_norm: np.ndarray = None, pc0: np.ndarray = None,
             sensor_dt: float = 0.1, data_name: str = 'av2'):
        """Compute metrics for one frame (matching eval.py's step_eval logic).
        
        Args:
            gt_dis: (N, 3) ground truth compensation distance.
            est_dis: (N, 3) estimated compensation distance.
            eval_mask: (N,) boolean mask for valid points.
            gt_category: (N,) category indices.
            gt_instance: (N,) instance IDs.
            gt_flow_norm: (N,) GT flow magnitude (for velocity filtering).
            pc0: (N, 3) original point cloud coordinates (for Chamfer calculation).
            sensor_dt: sensor time period (default 0.1s = 10Hz).
            data_name: dataset name for velocity threshold selection.
        """
        # Count this frame
        self.frame_cnt += 1

        # Apply eval mask to all arrays
        mask = eval_mask.astype(bool)
        gt_dis_masked = gt_dis[mask]
        est_dis_masked = est_dis[mask]
        gt_category_masked = gt_category[mask] if gt_category is not None else None
        gt_instance_masked = gt_instance[mask] if gt_instance is not None else None
        gt_flow_norm_masked = gt_flow_norm[mask] if gt_flow_norm is not None else None
        pc0_masked = pc0[mask] if pc0 is not None else None

        if gt_category_masked is None or gt_instance_masked is None:
            return

        # Velocity threshold for filtering slow-moving objects
        min_vel = 1.5 if data_name == 'scania' else 3.0

        # Per-frame score structure (matching eval.py's frame_score)
        def init_data():
            return {'num_pts': [], 'mpe': [], 'cham': []}
        frame_score = {
            "CAR": {'vel': {r: init_data() for r in ['0-10', '10-20', '20-30', '30+']}},
            "OTHER_VEHICLES": {'vel': {r: init_data() for r in ['0-10', '10-20', '20-30', '30+']}}
        }

        # Per-category, per-instance evaluation
        for cats_name in ["CAR", "OTHER_VEHICLES"]:
            classes_ids = [CATEGORY_TO_INDEX[cat] for cat in BUCKETED_METACATAGORIES[cats_name]]
            mask_class = np.isin(gt_category_masked, np.array(classes_ids))
            if not np.any(mask_class):
                continue

            gt_instance_class = gt_instance_masked[mask_class]
            gt_dis_class = gt_dis_masked[mask_class]
            est_dis_class = est_dis_masked[mask_class]
            gt_flow_norm_class = gt_flow_norm_masked[mask_class] if gt_flow_norm_masked is not None else None
            pc0_class = pc0_masked[mask_class] if pc0_masked is not None else None

            # Iterate instances
            for instance_id in np.unique(gt_instance_class):
                ins_mask = (gt_instance_class == instance_id)
                num_pts = np.sum(ins_mask)
                
                # Skip instances with < 10 points
                if num_pts < 10:
                    continue

                # Compute velocity for this instance
                if gt_flow_norm_class is not None:
                    vel_ins = np.mean(gt_flow_norm_class[ins_mask]) / sensor_dt
                else:
                    # If no flow norm available, skip velocity filtering
                    vel_ins = min_vel + 1  # Pass the filter
                
                # Skip slow-moving instances
                if vel_ins < min_vel:
                    continue

                # Compute MPE: mean L2 distance between gt and est compensation distance
                mpe = cal_mpe(gt_dis_class[ins_mask], est_dis_class[ins_mask])
                
                # Compute Chamfer: on refined point cloud (pc0 + comp_dis)
                if pc0_class is not None:
                    gt_refine_pc = pc0_class[ins_mask] + gt_dis_class[ins_mask]
                    est_refine_pc = pc0_class[ins_mask] + est_dis_class[ins_mask]
                    cham = cal_chamfer(gt_refine_pc, est_refine_pc)
                else:
                    cham = cal_chamfer(gt_dis_class[ins_mask], est_dis_class[ins_mask])

                # Categorize by velocity range (matching eval.py)
                if 0 < vel_ins < 10:
                    range_name = '0-10'
                elif 10 <= vel_ins < 20:
                    range_name = '10-20'
                elif 20 <= vel_ins < 30:
                    range_name = '20-30'
                elif vel_ins >= 30:
                    range_name = '30+'
                else:
                    continue  # Skip if velocity is invalid
                
                frame_score[cats_name]['vel'][range_name]['num_pts'].append(num_pts)
                frame_score[cats_name]['vel'][range_name]['mpe'].append(mpe)
                frame_score[cats_name]['vel'][range_name]['cham'].append(cham)

        # Aggregate frame results into global evaluate_data (matching eval.py)
        for cats_name in frame_score:
            total_num_list, mpe_list, cham_list = [], [], []
            
            for range_name in frame_score[cats_name]['vel']:
                num_pts_list = frame_score[cats_name]['vel'][range_name]['num_pts']
                if len(num_pts_list) > 0:
                    total_num_pts = sum(num_pts_list)
                    avg_mpe = np.average(frame_score[cats_name]['vel'][range_name]['mpe'], weights=num_pts_list)
                    avg_cham = np.average(frame_score[cats_name]['vel'][range_name]['cham'], weights=num_pts_list)
                    
                    # Store per-instance metrics to global data
                    self.evaluate_data[cats_name]['vel'][range_name]['num_pts'] += num_pts_list
                    self.evaluate_data[cats_name]['vel'][range_name]['mpe'] += frame_score[cats_name]['vel'][range_name]['mpe']
                    self.evaluate_data[cats_name]['vel'][range_name]['cham'] += frame_score[cats_name]['vel'][range_name]['cham']
                    
                    # Collect for frame-level mean
                    mpe_list.append(avg_mpe)
                    cham_list.append(avg_cham)
                    total_num_list.append(total_num_pts)

            # Update mean values (matching eval.py: nanmean across velocity ranges)
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
    
    def compute_scores(self) -> dict:
        """Compute final scores (matching eval.py's print function logic)."""
        
        def safe_average(values, weights):
            return np.average(values, weights=weights) if len(values) > 0 and np.sum(weights) > 0 else 0.0
        
        def safe_std(values):
            return np.std(values) if len(values) > 0 else 0.0

        ranges = ['0-10', '10-20', '20-30', '30+']

        # Per-category summaries
        cat_summary = {}
        for cat in ['CAR', 'OTHER_VEHICLES']:
            mean_raw = self.evaluate_data[cat]['mean']
            vel_raw = self.evaluate_data[cat]['vel']
            
            if len(mean_raw['num_pts']) == 0:
                cat_summary[cat] = {
                    'mpe_mean': 0.0, 'mpe_std': 0.0, 
                    'cham_mean': 0.0, 'cham_std': 0.0, 
                    'num_pts': 0, 'num_objs': 0,
                    'velocity': {r: {'mpe': 0.0, 'cd': 0.0, 'num_pts': 0, 'num_obj': 0} for r in ranges}
                }
                continue
            
            # Matching eval.py:
            # cat_mpe = safe_average(mean_raw['mpe'], mean_raw['num_pts'])
            # cat_std_mpe = safe_std(mean_raw['std_mpe'])
            mean_mpe = safe_average(mean_raw['mpe'], mean_raw['num_pts'])
            mean_cham = safe_average(mean_raw['cham'], mean_raw['num_pts'])
            std_mpe = safe_std(mean_raw['std_mpe'])
            std_cham = safe_std(mean_raw['std_cham'])
            npts = int(np.sum(mean_raw['num_pts']))
            nobj = len(mean_raw['num_pts'])  # number of frames with valid instances
            
            # Velocity breakdown (matching eval.py's savejson format)
            vel_breakdown = {}
            for r in ranges:
                v = vel_raw[r]
                vel_breakdown[r] = {
                    'mpe': float(safe_average(v['mpe'], v['num_pts'])),
                    'cd': float(safe_average(v['cham'], v['num_pts'])),
                    'num_pts': int(np.sum(v['num_pts'])) if len(v['num_pts']) > 0 else 0,
                    'num_obj': len(v['num_pts'])
                }
            
            cat_summary[cat] = {
                'mpe_mean': float(mean_mpe), 
                'mpe_std': float(std_mpe), 
                'cham_mean': float(mean_cham), 
                'cham_std': float(std_cham), 
                'num_pts': npts, 
                'num_objs': nobj,
                'velocity': vel_breakdown
            }

        # Compute overall weighted average across categories
        total_mpe_list = []
        total_cham_list = []
        total_pts_list = []
        
        for cat in ['CAR', 'OTHER_VEHICLES']:
            mean_raw = self.evaluate_data[cat]['mean']
            total_mpe_list.extend(mean_raw['mpe'])
            total_cham_list.extend(mean_raw['cham'])
            total_pts_list.extend(mean_raw['num_pts'])
        
        if len(total_pts_list) > 0:
            avg_mpe = safe_average(total_mpe_list, total_pts_list)
            avg_cham = safe_average(total_cham_list, total_pts_list)
        else:
            avg_mpe = 0.0
            avg_cham = 0.0
        
        return {
            # Overall metrics (weighted average)
            "mpe": float(avg_mpe),
            "chamfer": float(avg_cham),
            "num_frames": self.frame_cnt,
            "num_instances": int(len(total_pts_list)),
            "total_points": int(np.sum(total_pts_list)) if len(total_pts_list) > 0 else 0,
            # CAR category
            "car_cde": float(cat_summary['CAR']['cham_mean']),
            "car_mpe": float(cat_summary['CAR']['mpe_mean']),
            "car_num_objs": int(cat_summary['CAR']['num_objs']),
            "car_num_pts": int(cat_summary['CAR']['num_pts']),
            # OTHER_VEHICLES category
            "others_cde": float(cat_summary['OTHER_VEHICLES']['cham_mean']),
            "others_mpe": float(cat_summary['OTHER_VEHICLES']['mpe_mean']),
            "others_num_objs": int(cat_summary['OTHER_VEHICLES']['num_objs']),
            "others_num_pts": int(cat_summary['OTHER_VEHICLES']['num_pts']),
            # Keep nested structure for detailed analysis
            "per_category": cat_summary,
        }
    
    def save_detailed_json(self, data_name: str, flow_mode: str, file_path: str):
        """Save detailed metrics to JSON file matching eval.py's format.
        
        Args:
            data_name: Dataset name (e.g., 'av2', 'scania').
            flow_mode: Flow mode name (e.g., 'flow', 'seflowpp').
            file_path: Path to save the JSON file.
        """
        def safe_average(values, weights):
            return np.average(values, weights=weights) if len(values) > 0 and np.sum(weights) > 0 else 0.0
        
        def safe_std(values):
            return np.std(values) if len(values) > 0 else 0.0

        # Load existing file if present
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        if data_name not in data:
            data[data_name] = {}
        if flow_mode not in data[data_name]:
            data[data_name][flow_mode] = {}

        ranges = ['0-10', '10-20', '20-30', '30+']
        target_cats = ["CAR", "OTHER_VEHICLES"]

        for cat in target_cats:
            mean_raw = self.evaluate_data[cat]['mean']
            vel_raw = self.evaluate_data[cat]['vel']
            
            if len(mean_raw['num_pts']) == 0:
                continue

            # Overall stats for this category
            cat_mpe = safe_average(mean_raw['mpe'], mean_raw['num_pts'])
            cat_cd = safe_average(mean_raw['cham'], mean_raw['num_pts'])
            cat_std_mpe = safe_std(mean_raw['std_mpe'])
            cat_std_cd = safe_std(mean_raw['std_cham'])
            cat_n_pts = int(np.sum(mean_raw['num_pts']))
            cat_n_obj = len(mean_raw['num_pts'])

            # Build entry matching eval.py's savejson format
            entry = {
                "overall": {
                    "mpe": float(cat_mpe),
                    "cd": float(cat_cd),
                    "std_mpe": float(cat_std_mpe),
                    "std_cd": float(cat_std_cd),
                    "num_pts": cat_n_pts,
                    "num_obj": cat_n_obj
                },
                "velocity": {},
                "distance": {}  # Note: score.py doesn't track distance ranges, set to empty/zeros
            }

            # Velocity breakdown
            for r in ranges:
                v = vel_raw[r]
                entry["velocity"][r] = {
                    "mpe": float(safe_average(v['mpe'], v['num_pts'])),
                    "cd": float(safe_average(v['cham'], v['num_pts'])),
                    "num_pts": int(np.sum(v['num_pts'])) if len(v['num_pts']) > 0 else 0,
                    "num_obj": len(v['num_pts'])
                }
                # Distance not tracked in score.py (would need pc0 distance calculation)
                entry["distance"][r] = {
                    "mpe": 0.0,
                    "cd": 0.0,
                    "num_pts": 0,
                    "num_obj": 0
                }

            data[data_name][flow_mode][cat] = entry

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        
        return file_path


def score(gt_zip_path: str, pred_zip_path: str, output_dir: str = None, flow_mode: str = "submission") -> dict:
    """Main scoring function for CodaBench.
    
    Args:
        gt_zip_path: Path to ground truth zip file.
        pred_zip_path: Path to prediction zip file.
        output_dir: Optional output directory for scores.json.
        flow_mode: Name of the flow method (for detailed JSON file).
    
    Returns:
        Dictionary containing scores.
    """
    # Auto-detect dataset name from path
    if 'scania' in gt_zip_path.lower() or 'scania' in pred_zip_path.lower():
        data_name = 'scania'
    elif 'av2' in gt_zip_path.lower() or 'av2' in pred_zip_path.lower():
        data_name = 'av2'
    else:
        data_name = 'scania'  # default

    # List all sweeps in GT
    gt_sweeps = list_sweep_uuids(gt_zip_path)
    pred_sweeps = set(list_sweep_uuids(pred_zip_path))
    
    metrics = ScoreMetrics()
    
    for sweep_uuid in tqdm(gt_sweeps, desc="Scoring", ncols=80):
        if sweep_uuid not in pred_sweeps:
            # record missing prediction for reporting later
            if 'missing_preds' not in locals():
                missing_preds = []
            missing_preds.append(sweep_uuid)
            print(f"Warning: Missing prediction for {sweep_uuid}")
            continue

        gt_dis, eval_mask, gt_category, gt_instance, gt_flow_norm, pc0 = read_data_file(gt_zip_path, sweep_uuid)
        est_dis, *_ = read_data_file(pred_zip_path, sweep_uuid)

        if len(gt_dis) != len(est_dis):
            if 'mismatch_sweeps' not in locals():
                mismatch_sweeps = []
            mismatch_sweeps.append((sweep_uuid, len(gt_dis), len(est_dis)))
            print(f"Warning: Point count mismatch for {sweep_uuid}: GT={len(gt_dis)}, Pred={len(est_dis)}")
            continue

        metrics.step(gt_dis, est_dis, eval_mask, gt_category=gt_category, gt_instance=gt_instance, 
                     gt_flow_norm=gt_flow_norm, pc0=pc0, data_name=data_name)
    
    scores = metrics.compute_scores()
    
    # Print results
    print(f"\n{'='*50}")
    print(f"HiMo refinement metrics in {data_name}:")
    # Build table rows similar to eval.py
    cat = scores.get('per_category', {})
    def fmt(mean, std):
        return f"{mean:.3f} ± {std:.2f}" if std is not None else f"{mean:.3f}"

    total_mps = []
    total_chs = []
    total_pts = 0
    total_objs = 0
    table_rows = []
    for name in ['CAR', 'OTHER_VEHICLES']:
        c = cat.get(name, {})
        mpe_mean = c.get('mpe_mean', 0.0)
        mpe_std = c.get('mpe_std', 0.0)
        cham_mean = c.get('cham_mean', 0.0)
        cham_std = c.get('cham_std', 0.0)
        npts = c.get('num_pts', 0)
        nobj = c.get('num_objs', 0)
        table_rows.append([name if name!='OTHER_VEHICLES' else 'OTHERS', f"{cham_mean:.3f} ± {cham_std:.2f}", f"{mpe_mean:.3f} ± {mpe_std:.2f}", npts, nobj])
        total_pts += npts
        total_objs += nobj
        # collect per-instance lists if present
        total_mps += list(self_or_empty := (c.get('mpe_mean_list') if False else []))
        # we stored per-instance values in cat_data; but to avoid heavy restructuring, we will compute total from category means weighted by num_pts

    # Compute overall by weighting category means by points
    car = cat.get('CAR', {})
    oth = cat.get('OTHER_VEHICLES', {})
    def safe(x):
        return x if x is not None else 0.0
    # Weighted averages
    total_cham = 0.0
    total_mpe = 0.0
    if (car.get('num_pts',0) + oth.get('num_pts',0)) > 0:
        total_cham = (car.get('cham_mean',0.0)*car.get('num_pts',0) + oth.get('cham_mean',0.0)*oth.get('num_pts',0)) / max(1, car.get('num_pts',0) + oth.get('num_pts',0))
        total_mpe = (car.get('mpe_mean',0.0)*car.get('num_pts',0) + oth.get('mpe_mean',0.0)*oth.get('num_pts',0)) / max(1, car.get('num_pts',0) + oth.get('num_pts',0))

    # Insert Total row at top
    table_rows.insert(0, ['Total', f"{total_cham:.3f}", f"{total_mpe:.3f}", total_pts, total_objs])

    headers = ["Class", "CDE (Chamfer) ↓", "MPE (Point Err) ↓", "# Points", "# Objs"]
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid", stralign="center"))
    print(f"Total frames processed: {scores['num_frames']}")
    print(f"Results: printed above")
    print(f"{'='*50}\n")
    # Diagnostic: list missing or mismatched sweeps
    if 'missing_preds' in locals() and len(missing_preds) > 0:
        print(f"Missing predictions for {len(missing_preds)} sweeps. Examples:")
        print(missing_preds[:5])
    if 'mismatch_sweeps' in locals() and len(mismatch_sweeps) > 0:
        print(f"Point-count mismatches for {len(mismatch_sweeps)} sweeps. Examples (sweep, GT_count, Pred_count):")
        print(mismatch_sweeps[:5])
    
    # Save scores to output directory if provided (CodaBench format)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save CodaBench scores.json (required for leaderboard)
        scores_file = output_dir / "scores.json"
        with open(scores_file, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Scores saved to {scores_file}")
        
        # Save detailed JSON file matching eval.py format (for users who want more details)
        detailed_file = output_dir / f"res-{data_name}.json"
        metrics.save_detailed_json(data_name, flow_mode, str(detailed_file))
        print(f"Detailed results saved to {detailed_file}")
    
    return scores

def main():
    """Entry point for CodaBench or command line usage.
    
    CodaBench directory structure:
    - /app/input/ref/  : Ground truth data (reference_data from competition.yaml)
    - /app/input/res/  : User submission
    - /app/output/     : Output directory for scores.json
    
    The GT and submission can be zip files or extracted directories containing feather files.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="HiMo Benchmark Scoring Program")
    parser.add_argument("--gt_zip", type=str, default=None, help="Path to ground truth zip file or directory")
    parser.add_argument("--pred_zip", type=str, default=None, help="Path to prediction zip file or directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for scores.json")
    parser.add_argument("--flow_mode", type=str, default="submission", help="Name of the flow method for detailed JSON")
    args = parser.parse_args()
    
    # CodaBench mode: detect /app/input structure
    codabench_input = Path("/app/input")
    codabench_output = Path("/app/output")
    
    if codabench_input.exists() and args.gt_zip is None:
        # Running in CodaBench environment
        print("Detected CodaBench environment")
        
        ref_dir = codabench_input / "ref"
        res_dir = codabench_input / "res"
        
        # Debug: print directory contents (limited)
        print(f"ref_dir exists: {ref_dir.exists()}")
        print(f"res_dir exists: {res_dir.exists()}")
        
        # Find GT data path
        gt_path = None
        
        # Pattern 1: Direct zip file in ref/
        gt_zips = list(ref_dir.glob("*.zip"))
        if gt_zips:
            gt_path = str(gt_zips[0])
            print(f"Found GT zip: {gt_path}")
        
        # Pattern 2: Dataset was extracted, use ref_dir directly
        if gt_path is None:
            feather_files = list(ref_dir.rglob("*.feather"))
            if feather_files:
                print(f"Found {len(feather_files)} feather files (GT is extracted)")
                gt_path = str(ref_dir)
                print(f"Using extracted GT directory: {gt_path}")
        
        if gt_path is None:
            raise FileNotFoundError(f"No GT data found in {ref_dir}")
        
        # Find prediction data path
        pred_path = None
        
        # Pattern 1: Direct zip file in res/
        pred_zips = list(res_dir.glob("*.zip"))
        if pred_zips:
            pred_path = str(pred_zips[0])
            print(f"Found prediction zip: {pred_path}")
        
        # Pattern 2: Submission was extracted, use res_dir directly
        if pred_path is None:
            feather_files = list(res_dir.rglob("*.feather"))
            if feather_files:
                print(f"Found {len(feather_files)} feather files (submission is extracted)")
                pred_path = str(res_dir)
                print(f"Using extracted prediction directory: {pred_path}")
        
        if pred_path is None:
            raise FileNotFoundError(f"No prediction data found in {res_dir}")
        
        output_dir = str(codabench_output)
        flow_mode = "submission"  # default for CodaBench
    else:
        # Command line mode
        if args.gt_zip is None or args.pred_zip is None:
            parser.error("--gt_zip and --pred_zip are required when not running in CodaBench")
        gt_path = args.gt_zip
        pred_path = args.pred_zip
        output_dir = args.output_dir
        flow_mode = args.flow_mode
    
    score(gt_path, pred_path, output_dir, flow_mode)


if __name__ == "__main__":
    main()