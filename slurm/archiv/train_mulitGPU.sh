#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=alpha
#SBATCH --gres=gpu:8
#SBATCH --job-name=DIFFUSION-multiGPU_%j
#SBATCH --output=slurm-out/diffusion_train-multiGPU-%j.out    # after the shebang line
#SBATCH --mail-type=begin,end,FAIL,TIME_LIMIT
#SBATCH --mail-user=christian.duereth@tu-dresden.de

module load modenv/hiera  GCC/10.2.0  CUDA/11.7.1
module load Python/3.8.6
module load OpenMPI/4.0.5
## just to check if GPUs are available
nvidia-smi

source /home/h2/dchristi/alpha_Py386_CU116/bin/activate
#################################################################################
## Settings for the model
#################################################################################
## TODO add in_channel 
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True --num_classes=3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
NUM_GPUS=8
#################################################################################
CHECK_FLAGS="--resume_checkpoint=/lustre/ssd/ws/dchristi-diffusion2/checkpoints/fiber_1x256x256_cond3_1k/model280000.pt"
TRAIN_FLAGS="--lr 1e-4 --batch_size 1 --lr_anneal_steps=300000"
DIR_TRAIN="--data_dir /lustre/ssd/ws/dchristi-diffusion2/dataset/fiber-data_256x256 --dir /lustre/ssd/ws/dchristi-diffusion2/checkpoints/fiber_1x256x256_cond3_1k"
#################################################################################


mpiexec -n $NUM_GPUS python scripts/image_train.py $DIR_TRAIN $MODEL_FLAGS $CHECK_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


exit 0