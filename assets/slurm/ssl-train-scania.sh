#!/bin/bash
#SBATCH -J seflowpp
#SBATCH --gpus 4 -C "thin"
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-364/users/x_qinzh/workspace/OpenSceneFlow/logs/slurm/%J.out
#SBATCH --error  /proj/berzelius-2023-364/users/x_qinzh/workspace/OpenSceneFlow/logs/slurm/%J.err

PYTHON=/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/seflow_v2/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib
cd /proj/berzelius-2023-364/users/x_qinzh/workspace/OpenSceneFlow

# ===> Need change it data path changed <===
SOURCE="/proj/berzelius-2023-154/users/x_qinzh"
DEST="/scratch/local"
SUBDIRS=("pdata/train" "pdata/val")

start_time=$(date +%s)
for dir in "${SUBDIRS[@]}"; do
    mkdir -p "${DEST}/${dir}"
    find "${SOURCE}/${dir}" -type f -print0 | xargs -0 -n1 -P16 cp -t "${DEST}/${dir}" &
done
wait
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Copy ${SOURCE} to ${DEST} Total time: ${elapsed} seconds"
echo "Start training..."

# # ========================= SeFlow++ (HiMo) =========================
$PYTHON train.py slurm_id=$SLURM_JOB_ID wandb_mode=online wandb_project_name=himorain_data=/scratch/local/pdata/train val_data=/scratch/local/pdata/val  \
     model=deflowpp save_top_model=3 val_every=3 "voxel_size=[0.2, 0.2, 6]" "point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3]" \
     num_workers=16 epochs=15 lr=1e-4 batch_size=8 "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
     +ssl_label=seflowpp_auto loss_fn=seflowppLoss num_frames=3