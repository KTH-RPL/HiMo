"""
Repack Scania h5 files:
1. Rename 'SensorsCenter' to 'lidar_center'
2. Change lidar_center to (N, 4, 4) with identity rotation
3. Delete 'himu_seflowpp' key
4. Fix data types for PyTorch compatibility (uint32 -> int64, etc.)
5. Repack the h5 file to reclaim space

Usage:
    python tools/repack_h5_scania.py --data_dir /home/kin/data/scania/val_v1repacked
"""

import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil

# Define expected dtypes for each key (PyTorch compatible)
# Supported types: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, bool
DTYPE_MAP = {
    # 'timestamp': np.int64,
    # 'flow_instance_id': np.int32,  # uint32 -> int32
    'lidar_id': np.uint8,
    # 'flow_category_indices': np.uint8,
    # 'flow_is_valid': np.bool_,
    'ground_mask': np.bool_,
    'lidar': np.float32,
    'lidar_dt': np.float32,
    # 'flow': np.float32,
    'pose': np.float64,
    'ego_motion': np.float32,
    'lidar_center': np.float32,
}


def process_single_h5(h5_path: Path, output_path: Path):
    """Process a single h5 file and save to output path."""
    
    with h5py.File(h5_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            for timestamp_key in f_in.keys():
                group_in = f_in[timestamp_key]
                group_out = f_out.create_group(timestamp_key)
                
                for sub_key in group_in.keys():
                    # Skip himu_seflowpp
                    if sub_key in ['seflowpp_best']:
                        continue
                    
                    # Rename SensorsCenter to lidar_center and convert to (N, 4, 4)
                    # Also handle existing lidar_center that needs to be converted
                    if sub_key == 'SensorsCenter' or sub_key == 'lidar_center':
                        data = group_in[sub_key][:]
                        # Convert to (N, 4, 4) with identity rotation and translation from original data
                        if len(data.shape) == 3 and data.shape[1] == 4 and data.shape[2] == 4:
                            # Already (N, 4, 4), keep as is
                            new_data = data.astype(np.float32)
                        elif len(data.shape) == 2 and data.shape[1] == 3:
                            # (N, 3) -> (N, 4, 4) with identity rotation
                            N = data.shape[0]
                            new_data = np.zeros((N, 4, 4), dtype=np.float32)
                            new_data[:, :3, :3] = np.eye(3)  # identity rotation
                            new_data[:, :3, 3] = data  # translation
                            new_data[:, 3, 3] = 1.0
                        else:
                            print(f"Warning: Unexpected shape {data.shape} for {sub_key} in {h5_path}/{timestamp_key}")
                            new_data = data.astype(np.float32)
                        
                        group_out.create_dataset('lidar_center', data=new_data)
                    else:
                        if sub_key not in DTYPE_MAP:
                            print(f"Warning: {sub_key} not in DTYPE_MAP, skip this key.")
                            continue
                        # Copy other datasets with proper dtype conversion
                        # Handle scalar datasets (e.g., timestamp)
                        if group_in[sub_key].shape == ():
                            data = group_in[sub_key][()]
                        else:
                            data = group_in[sub_key][:]
                        
                        # Convert dtype if needed for PyTorch compatibility
                        if sub_key in DTYPE_MAP:
                            data = np.array(data, dtype=DTYPE_MAP[sub_key])
                        elif data.dtype == np.uint32:
                            # uint32 not supported by PyTorch, convert to int64
                            data = data.astype(np.int64)
                        elif data.dtype == np.uint64:
                            # uint64 not supported by PyTorch, convert to int64
                            data = data.astype(np.int64)
                        
                        group_out.create_dataset(sub_key, data=data)


def main(data_dir: str, in_place: bool = True):
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: {data_path} does not exist")
        return
    
    h5_files = sorted(list(data_path.glob("*.h5")))
    print(f"Found {len(h5_files)} h5 files in {data_path}")
    
    if len(h5_files) == 0:
        print("No h5 files found!")
        return
    
    # Create temp directory for repacked files
    temp_dir = data_path / "_temp_repack"
    temp_dir.mkdir(exist_ok=True)
    
    for h5_file in tqdm(h5_files, desc="Processing h5 files"):
        temp_output = temp_dir / h5_file.name
        
        try:
            process_single_h5(h5_file, temp_output)
            
            if in_place:
                # Replace original file with repacked one
                shutil.move(str(temp_output), str(h5_file))
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            if temp_output.exists():
                temp_output.unlink()
            continue
    
    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repack Scania h5 files")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to directory containing h5 files")
    parser.add_argument("--no_in_place", action="store_true",
                        help="If set, don't replace original files (for testing)")
    
    args = parser.parse_args()
    main(args.data_dir, in_place=not args.no_in_place)
