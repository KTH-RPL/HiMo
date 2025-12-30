import pickle as pkl
import fire, os, sys


def main(
    full_pkl_path: str = "/home/kin/data/av2/h5py/sensor/himo/index_total.pkl",
    new_folder: str = "/home/kin/data/av2/h5py/sensor/himo/demo"
):
    with open(full_pkl_path, 'rb') as f:
        data = pkl.load(f)
    # read the scene_id.h5 file under new folder
    scene_ids = [file_name.split('.')[0] for file_name in os.listdir(new_folder) if file_name.endswith('.h5')]
    new_pkl_content = []
    for scene_id, timestamps in data:
        if scene_id in scene_ids:
            new_pkl_content.append([scene_id, timestamps])

    with open(os.path.join(new_folder, full_pkl_path.split('/')[-1]), 'wb') as f:
        pkl.dump(new_pkl_content, f)

if __name__ == '__main__':
    fire.Fire(main)