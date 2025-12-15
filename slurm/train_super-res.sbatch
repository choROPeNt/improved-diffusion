#!/bin/bash

#Submit this script with: sbatch thefilename

#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=10:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=6                               # number of processor cores (i.e. threads)
#SBATCH --partition=capella
#SBATCH --mem-per-cpu=12G                               # memory per CPU core
#SBATCH --gres=gpu:1                                    # number of gpus
#SBATCH -J "Diff_super-res"                             # job name
#SBATCH --output=out/Diff_super-res-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_biiax
##################################################################
##################################################################



ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0 OpenMPI/5.0.3


source /data/horse/ws/dchristi-diffusion/.venv/bin/activate
export OMPI_MCA_smsc=^knem

echo "Running with ${SLURM_NTASKS} GPU(s)"

mpiexec -n ${SLURM_NTASKS} python scripts/super_res_train.py \
    --data_dir "/data/horse/ws/dchristi-diffusion/dataset/Thinlayer 2-2_2023-04-28_124148/patches" \
    --dir "/data/horse/ws/dchristi-diffusion/checkpoints/super-res-256x256_4x" \
#     # --image_size 256 \
#     # --lr 1e-4 \
#     # --batch_size 4 \
#     # --num_channels 128 \
#     # --num_res_blocks 3 \
#     # --learn_sigma True \
#     # --class_cond True \
#     # --num_classes 3 \
#     # --diffusion_steps 1000 \
#     # --noise_schedule linear \
#     # --lr_anneal_steps=500000