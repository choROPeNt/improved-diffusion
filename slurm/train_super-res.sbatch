#!/bin/bash

#Submit this script with: sbatch thefilename

#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=12:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=4                                      # limit to one node
#SBATCH --cpus-per-task=6                               # number of processor cores (i.e. threads)
#SBATCH --partition=capella
#SBATCH --mem-per-cpu=8G                               # memory per CPU core
#SBATCH --gres=gpu:4                                    # number of gpus
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

nvidia-smi

echo "Job CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

mpiexec -n ${SLURM_NTASKS} bash -lc '
  echo "rank=$OMPI_COMM_WORLD_RANK local_rank=$OMPI_COMM_WORLD_LOCAL_RANK host=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"
  python - << "PY"
import os, torch
print(" torch cuda visible:", torch.cuda.device_count())
print(" current device:", torch.cuda.current_device())
print(" device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "no cuda")
PY
'


mpiexec -n ${SLURM_NTASKS} bash -lc '
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  exec python scripts/super_res_train.py \
    --data_dir "/data/horse/ws/dchristi-diffusion/dataset/Thinlayer 2-2_2023-04-28_124148/patches" \
    --dir "/data/horse/ws/dchristi-diffusion/checkpoints/super-res-256x256_2x" \
    --large_size 256 \
    --small_size 64 \
    --lr 1e-4 \
    --batch_size 4 \
    --num_channels 192 \
    --num_res_blocks 2 \
    --learn_sigma True \
    --diffusion_steps 4000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False
'
