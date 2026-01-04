Downstream Tasks
---

## 3D Detection

We use [OpenPCDet](https://github.com/Kin-Zhang/OpenPCDet) for the 3D detection task. Please reference the official repo for env setup and model pretrained weights download.

The pretrained model we use in the paper is [TransFusion-L*](https://drive.google.com/file/d/1cuZ2qdDnxSwTCsiXWwbqCGF-uoazTXbz/view?usp=share_link). Downloaded link from Official OpenPCDet repo.

Then we run on our Scania validation dataset (Raw w. ego-motion compensation and w. HiMo compensated results), you can run the following command under `downstream/OpenPCDet` folder to:

```bash
cd OpenPCDet/tools
python3 h5sf.py --cfg_file cfgs/nuscenes_models/transfusion_lidar.yaml \
    --ckpt /home/kin/model_zoo/cbgs_transfusion_lidar.pth \
    --data_path /home/kin/data/scania/val --vis True \
    --flow_mode 'seflowpp_best' 
```

Demo results, the example scene in the paper (Fig. 12) is from `batch_184_20211217173504`.
<!-- ![Detection Results](assets/docs/himo_seflowpp_scania_det.png) -->

## Segmentation

We use [WaffleIron](https://github.com/Kin-Zhang/WaffleIron) for the segmentation task in Argoverse 2 dataset (as the intensity can be normalized to 0-1 and similar to original WaffleIron training setting).

1. Run seg model method and write the result into h5 file.
```bash
cd WaffleIron
python eval_h5.py \
--path_dataset /home/kin/data/av2/h5py/sensor/himo \
--ckpt ./pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--phase test --flow_mode raw 

python eval_h5.py \
--path_dataset /home/kin/data/av2/h5py/sensor/himo \
--ckpt ./pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--phase test --flow_mode seflowpp_best
```

- phase: val (for only validation frame, for quantative results) or test (for all frames in the seq, mainly for visulization)
- source: raw or himo (distorted or undistorted data, himo directly use seflow++ version.)


Then you can get the segmentation mIoU table as shown in the paper (Table IV):
```bash
python eval_seg.py \
--data_dir /home/kin/data/av2/h5py/sensor/himo \
--res_names "['seg_raw', 'seg_seflowpp_best']"
```


| Input Point Cloud    | Mask Only |         | All     |        |
| :------------------- | :-------: | :-----: | :-----: | :----: |
|                      | CAR       | OTHER.  | CAR     | OTHER. |
| w. ego-motion comp.  | 80\.909   | 31\.44  | 66\.081 | 9\.837 |
| w. HiMo motion comp. | 81\.541   | 35\.398 | 66\.438 | 11\.15 |

More detail about the item: 
> The model is trained on KITTI - an urban dataset with mainly low-speed scenes and therefore low distortion. We then apply WaffleIron to the high-speed Argoverse 2 validation frames, comparing two input variants: (i) raw point clouds with baseline ego-motion compensation and (ii) point clouds corrected by HiMo. To ensure a fair evaluation across different sensor setups and annotation coverage, we report two sets of IoU scores. "All" reports the IoU over all points, whilst "Mask only" reports the IoU over points that fall into the labelled ground-truth bounding boxes.