#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --account=p_autoshear
#SBATCH --mem-per-cpu=40000
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --job-name=diff_test
#SBATCH --output=testout.log
#SBATCH --open-mode=append
#SBATCH --mail-type=begin,end,FAIL,TIME_LIMIT
#SBATCH --mail-user=paul.seibert@tu-dresden.de

# conda deactivate

module load modenv/hiera  GCC/10.2.0  CUDA/11.7
# module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1
module load Python/3.8.6
module load OpenMPI/4.0.5

source /lustre/ssd/ws/dchristi-diffusion2/.venv/bin/activate

python scripts/image_sample_class.py --model_path /lustre/ssd/ws/dchristi-diffusion2/dataset/checkpoints/ema_0.9999_058000.pt --diffusion_steps 500 --image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma FALSE --num_classes 3 
# python scripts/image_sample_class.py --model_path /lustre/ssd/ws/dchristi-diffusion2/dataset/checkpoints/opt058000.pt --diffusion_steps 500 --image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma FALSE --num_classes 3 

exit 0
