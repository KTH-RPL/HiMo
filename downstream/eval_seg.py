
"""
# Created: 2025-03-24 11:39
# Copyright (C) 2025-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: view scene flow dataset after preprocess.
"""

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
import time, fire, h5py, pickle
import numpy as np

from av2.datasets.sensor.constants import AnnotationCategories
from typing import Final

from tqdm import tqdm

CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}
INDEX_TO_CATEGORY: Final = {v: k for k, v in CATEGORY_TO_INDEX.items()}
NAME_MAPPING_K2A = {
    'outlier': 'NONE',
    'unlabeled': 'NONE',
    'car': 'REGULAR_VEHICLE',
    'bicycle': 'BICYCLE',
    'motorcycle': 'MOTORCYCLE',
    'truck': 'TRUCK',
    'other-vehicle': 'LARGE_VEHICLE',
    'person': 'PEDESTRIAN',
    'bicyclist': 'BICYCLIST',
    'motorcyclist': 'MOTORCYCLIST',
    'road': 'NONE',
    'parking': 'NONE',
    'sidewalk': 'NONE',
    'other-ground': 'NONE',
    'building': 'NONE',
    'fence': 'NONE',
    'vegetation': 'NONE',
    'trunk': 'NONE',
    'terrain': 'NONE',
    'pole': 'NONE',
    'traffic-sign': 'SIGN',
}


NAME_MAPPING_N2A = {
    'ignore': 'NONE',
    'barrier': 'NONE',
    'bicycle': 'BICYCLE',
    'bus': 'BUS',
    'car': 'REGULAR_VEHICLE',
    'construction_vehicle': 'LARGE_VEHICLE',
    'motorcycle': 'MOTORCYCLE',
    'pedestrian': 'PEDESTRIAN',
    'traffic_cone': 'NONE',
    'trailer': 'VEHICULAR_TRAILER',
    'truck': 'TRUCK',
    'driveable_surface': 'NONE',
    'other_flat': 'NONE',
    'sidewalk': 'NONE',
    'terrain': 'NONE',
    'manmade': 'NONE',
    'vegetation': 'NONE',
}
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
class iouEval:
  def __init__(self, n_classes=2, ignore=None):
    # classes
    self.n_classes = n_classes
    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
    # print("[IOU EVAL] IGNORE: ", self.ignore)
    # print("[IOU EVAL] INCLUDE: ", self.include)
    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

  def addBatch(self, x, y):  # x=preds, y=targets
    # to tensor
    x_row = x.astype(np.int64)
    y_row = y.astype(np.int64)

    # sizes should be matching
    x_row = x_row.reshape(-1)  # de-batchify
    y_row = y_row.reshape(-1)  # de-batchify

    # check
    assert(x_row.shape == x_row.shape)

    # idxs are labels and predictions
    idxs = np.stack([x_row, y_row], axis=0)

    # ones is what I want to add to conf when I
    ones = np.ones((idxs.shape[-1]), dtype=np.int64)

    # make confusion matrix (cols = gt, rows = pred)
    # self.conf_matrix = self.conf_matrix.index_put_(
    #     tuple(idxs), ones, accumulate=True)
    np.add.at(self.conf_matrix, tuple(idxs), ones)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.astype(np.float64)
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES
  
class HDF5Data:
    def __init__(self, directory, flow_view=False, vis_name=["flow"], val=True):
        '''
        directory: the directory of the dataset
        t_x: how many past frames we want to extract
        '''
        self.flow_view = flow_view
        self.vis_name = vis_name if isinstance(vis_name, list) else [vis_name]
        self.directory = directory
        self.phase = 'val' if val else 'test'
        if os.path.exists(os.path.join(self.directory, 'index_eval.pkl')) or self.phase == 'val':
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            with open(eval_index_file, 'rb') as f:
                self.evalim_idx = pickle.load(f)
        else:
            eval_index_file = None
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.phase == 'val':
            return len(self.evalim_idx)
        return len(self.data_index)
    
    def __getitem__(self, index):
        if self.phase == 'val':
            scene_id, timestamp = self.evalim_idx[index]
            index = self.data_index.index([scene_id, timestamp])
        scene_id, timestamp = self.data_index[index]
        # to make sure we have continuous frames for flow view
        # if self.flow_view and self.scene_id_bounds[scene_id]["max_index"] == index:
        #     index = index - 1
        #     scene_id, timestamp = self.data_index[index]

        key = str(timestamp)
        data_dict = {
            'scene_id': scene_id,
            'timestamp': timestamp,
        }
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            # original data
            data_dict['pc0'] = f[key]['lidar'][:]
            data_dict['gm0'] = f[key]['ground_mask'][:]
            data_dict['pose0'] = f[key]['pose'][:]
            for flow_key in ['seg_valid', 'flow_category_indices'] + self.vis_name:
                if flow_key in f[key]:
                    data_dict[flow_key] = f[key][flow_key][:]
                else:
                    print(f"[Warning]: No {flow_key} in {scene_id} at {timestamp}, check the data.")
            # if self.flow_view:
            #     next_timestamp = str(self.data_index[index+1][1])
            #     data_dict['pose1'] = f[next_timestamp]['pose'][:]
            #     data_dict['pc1'] = f[next_timestamp]['lidar'][:]
            #     data_dict['gm1'] = f[next_timestamp]['ground_mask'][:]
            # elif self.flow_view:
            #     print(f"[Warning]: No {self.vis_name} in {scene_id} at {timestamp}, check the data.")
        return data_dict

valid_index_ = [CATEGORY_TO_INDEX[l] for l in CAR + OTHER_VEHICLES]
def main(
    data_dir: str ="/home/kin/data/av2/h5py/sensor/himo",
    res_names: list = ["seg_raw","seg_flow"]
):
    dataset = HDF5Data(data_dir, flow_view=True, vis_name=res_names, val=True)
    evaluators = {name: iouEval(n_classes=3, ignore=[]) for name in res_names}
    # print(f"Total {len(dataset)} scenes.")
    car_index = [CATEGORY_TO_INDEX[l] for l in CAR]
    other_index = [CATEGORY_TO_INDEX[l] for l in OTHER_VEHICLES]
    for data_id in tqdm(range(len(dataset)), desc="Evaluating", total=len(dataset), ncols=120):
        data = dataset[data_id]
        if 'flow_category_indices' not in data:
            print(f"[Warning]: No flow_category_indices in {data['scene_id']} at {data['timestamp']}, check the data.")
            continue
        valid_mask = data['seg_valid']
        # mask only or all points
        valid_mask = np.ones_like(valid_mask)
        seg_gt = data['flow_category_indices'][valid_mask]
        
        # re-assign the label needed on 3 classes only
        # if not car_index and other_index, then it is 0
        seg_gt[~np.isin(seg_gt, valid_index_)] = 0
        seg_gt[np.isin(seg_gt, car_index)] = 1
        seg_gt[np.isin(seg_gt, other_index)] = 2
        seg_pred = {}
        for name in res_names:
            seg_pred[name] = data[name][valid_mask]
            seg_pred[name][~np.isin(seg_pred[name], valid_index_)] = 0
            seg_pred[name][np.isin(seg_pred[name], car_index)] = 1
            seg_pred[name][np.isin(seg_pred[name], other_index)] = 2
            
            evaluators[name].addBatch(seg_pred[name], seg_gt)

        # if data_id > 10:
        #     break


        # evaluate miou on valid mask only and maybe highspeed? 
    print("\n  ========================== RESULTS ==========================  ")
    for name in res_names:
        _, class_jaccard = evaluators[name].getIoU()
        m_jaccard = class_jaccard[1:].mean()

        ignore = [0]
        class_strings = {0:'ignore', 1: 'car', 2: 'other_vehicle'}
        print('{name} 100 frames val:\nIoU avg {m_jaccard:.3f}'.format(name=name, m_jaccard=m_jaccard*100))
        # print also classwise
        for i, jacc in enumerate(class_jaccard):
            if i in ignore:
                continue
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[i], jacc=jacc*100))
        print('-'*20)
if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")