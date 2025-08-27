import pickle
import os, sys

import h5py
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)

class HDF5Data:
    def __init__(self, directory, res_name=None, eval=False):
        '''
        directory: the directory of the dataset
        t_x: how many past frames we want to extract
        '''

        self.res_name = res_name
        self.directory = directory
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
        
        self.eval_index = False
        if eval:
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            if not os.path.exists(eval_index_file):
                # raise Exception(f"No eval index file found! Please check {self.directory}")
                eval_index_file = os.path.join(BASE_DIR, 'assets/docs/index_eval_v2.pkl')
            self.eval_index = eval
            with open(eval_index_file, 'rb') as f:
                self.eval_data_index = pickle.load(f)
                
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
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp])
        else:
            scene_id, timestamp = self.data_index[index]
            # to make sure we have continuous frames for flow view
            if self.flow_view and self.scene_id_bounds[scene_id]["max_index"] == index:
                self.__getitem__(index-1)

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            data_dict = {
                'scene_id': scene_id,
                'timestamp': timestamp,
                
                'lidar_id': f[key]['lidar_id'][:] if 'lidar_id' in f[key] else None,
                'dt0': f[key]['lidar_dt'][:] if 'lidar_dt' in f[key] else None,
                'SensorsCenter': f[key]['SensorsCenter'][:] if 'SensorsCenter' in f[key] else None,
            }
            data_dict['pc0'] = f[key]['lidar'][:]
            data_dict['gm0'] = f[key]['ground_mask'][:]
            data_dict['pose0'] = f[key]['pose'][:]
            # flow results etc.
            if self.res_name in f[key]:
                data_dict[self.res_name] = f[key][self.res_name][:]
            next_timestamp = str(self.data_index[index+1][1])
            data_dict['pose1'] = f[next_timestamp]['pose'][:]
            data_dict['pc1'] = f[next_timestamp]['lidar'][:]
            data_dict['gm1'] = f[next_timestamp]['ground_mask'][:]

        return data_dict