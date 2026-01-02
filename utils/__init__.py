import numpy as np
import os, sys

def check_valid(data_dir, flow_mode, comp_dis_zip):    
    # Different dataset have diff ego car size.
    if data_dir.find("Scania") > 0 or data_dir.find("scania") > 0:
        data_name = "scania"
    elif data_dir.find("av2") > 0 or data_dir.find("AV2") > 0:
        data_name = "av2"
    else:
        raise ValueError("Unknown dataset name in data_dir.")

    EVAL_FLAG = 0 # invalid
    # check whether the comp_dis_zip is provided.
    if os.path.exists(comp_dis_zip):
        print(f"Using provided comp_dis_zip: {comp_dis_zip} for evaluation.")
        EVAL_FLAG = 1
    else:
        print(f"No valid comp_dis_zip provided, evaluating based on {flow_mode} directly.")
        
        # TODO: check whether flow_mode exists in the data.
        EVAL_FLAG = 2

    return data_name, EVAL_FLAG

def ego_pts_mask(pts, min_bound=[-9.5, -3/2, 0], max_bound=[5, 2.760004/2, 5]):
    """
    Input: pts: (N, 3); min_bound: (3, ); max_bound: (3, )
    Output: mask: (N, ) indicate the points that are outside the ego vehicle box.
    """
    mask = ((pts[:, 0] > min_bound[0]) & (pts[:, 0] < max_bound[0])
            & (pts[:, 1] > min_bound[1]) & (pts[:, 1] < max_bound[1])
            & (pts[:, 2] > min_bound[2]) & (pts[:, 2] < max_bound[2]))
    return ~mask

def flow2compDis(flow, dt0, sensor_dt=10):
    """
    refine points belong to a instance based on the flow and dt0.
    dt0: nanosecond offsets _from_ the start of the sweep. 
        * The first point dt is normally 0.0 since it's the start of the sweep.
        * So the latest point dt is normally T_sensor like 0.1.
    """
    return flow/sensor_dt * dt0[:, None] 

def refine_pts(pc, ds):
    ref_pc = pc[:,:3] + ds
    return ref_pc