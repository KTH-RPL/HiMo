HiMo: High-Speed Objects Motion Compensation in Point Clouds
---

[![arXiv](https://img.shields.io/badge/arXiv-2503.00803-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.00803)
[![page](https://img.shields.io/badge/Project-Page-green)](https://kin-zhang.github.io/HiMo)
[![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/rofaKfezIx0?si=59mMPLYUMgvrkRGj)

Note: I knew sometime we might want to see codes asap, so I upload all my experiment codes without cleaning up (some lib might missing etc). I will try my best to cleanup TBD list here:

- [x] Update the repo README.
- [x] Update OpenSceneFlow repo for dataprocess and SeFlow++.
- [ ] Test successfully all codes in this repo.
- [ ] Upload Scania validation set (w/o gt).
- [ ] Setup leaderboard for users get their Scania val score.
- [ ] Downstream task two repos README update.
- [ ] Public the author-response file for readers to check some discussion and future directions etc.

## Environment Setup

Please refer to [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow) to setup the `opensf` Python environment.

## Run HiMo

For HiMo paper (with review discussion), we discussed two possible way to do motion compensation:
1) In our paper, we propose HiMo, a modular pipeline that repurposes scene flow to compute distortion correction vectors for each point.
2) While it is theoretically possible to attempt such correction using alternative approaches (e.g.,dynamic segmentation + ICP alignment across LiDARs), such solutions would require complex ad-hoc heuristics for target selection, tracking, and optimization and have not been shown to work in practice.

We present Our HiMo pipeline **how to use scene flow** for motion compensation here:

### Run Scene Flow

Please refer to [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow) for output the flow, you can use SeFlow++ or other optimization-based baselines in our paper, we pushed all baseline code into our codebase also.

After training, please run [OpenSceneFlow/save.py] for output the flow results:
```bash
# (feed-forward): load ckpt
python save.py checkpoint=/home/kin/model_zoo/seflow_ppbest.ckpt dataset_path=/home/kin/data/scania/val

# (optimization-based): change another model by passing model name.
python save.py model=nsfp dataset_path=/home/kin/data/av2/h5py/demo/val
```

### Save Motion Compensation

The methods in HiMo, we first run the scene flow then save the compensation result by [save.py]. While for local validation like in av2, you can directly run [eval.py] as we support read flow and then compensated inside eval code, check more in [evaluation section](#evaluation).
```bash
python save.py
```

### Downstream Task

In the paper, we present Segmentation Task: [WaffleIron](https://github.com/KTH-RPL/WaffleIron) and 3D Detection Task: [OpenPCDet](https://github.com/Kin-Zhang/OpenPCDet/tree/feature/himo). 
We add the running README in each repo for users to be able run HiMo results with downstream task.


## Evaluation

The methods in HiMo, we directly read the scene flow result and output the scores.

```bash
python eval.py
```


## Visualization

To visualize the HiMo results, we refer OpenSceneFlow open3d visualizer to do so:
```bash
python visualize.py --flow "['raw', 'seflowpp', 'nsfp']"
```

For paper results, I manually select some instance for a clearer qualitative comparison etc.
```bash
python tools/view_instance.py
```

For project website animation, check [tools/animation_video.py](tools/animation_video.py).
```python
# first step:
fire.Fire(save_animation_traj)

# second step:
fire.Fire(animation_video)
```

For Video animation example, we use [manim](https://www.manim.community/). I may upload this part also for tutorial purpose etc.

## Cite & Acknowledgements

```
@article{zhang2025himo,
    title={{HiMo}: High-Speed Objects Motion Compensation in Point Clouds},
    author={Zhang, Qingwen and Khoche, Ajinkya and Yang, Yi and Ling, Li and Sina, Sharif Mansouri and Andersson, Olov and Jensfelt, Patric},
    year={2025},
    journal={arXiv preprint arXiv:2503.00803},
}
```

💞 Thanks to Bogdan Timus and Magnus Granström from Scania and Ci Li from KTH RPL, who helped with this work. 
We also thank Yixi Cai, Yuxuan Liu, Peizheng Li and Shenghai Yuan for helpful discussions during revision.
We also thank the anonymous reviewers for their useful comments.

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and Prosense (2020-02963) funded by Vinnova. 
The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.