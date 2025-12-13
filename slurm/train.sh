#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --account=p_autoshear
#SBATCH --mem-per-cpu=24000
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --job-name=DIFFUSION_%j
#SBATCH --output=diffusion_train-%j.out    # after the shebang line
#SBATCH --mail-type=begin,end,FAIL,TIME_LIMIT
#SBATCH --mail-user=paul.seibert@tu-dresden.de

module load modenv/hiera  GCC/10.2.0  CUDA/11.7.0
module load Python/3.8.6
module load OpenMPI/4.0.5

source /lustre/ssd/ws/dchristi-diffusion2/.venv_gpu/bin/activate
# source /lustre/ssd/ws/dchristi-diffusion2/.venv/bin/activate

## Settings for the model
## TODO add in_channel 
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma FALSE --class_cond True --num_classes=3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"


# CHECK_FLAGS="--resume_checkpoint=/lustre/ssd/ws/dchristi-diffusion2/checkpoints/fiber_1x256x256_cond3_1k/model250000.pt"
TRAIN_FLAGS="--lr 3e-5 --batch_size 8 --lr_anneal_steps=300000 --log_interval 10000"
DIR_TRAIN="--data_dir /lustre/ssd/ws/dchristi-diffusion2/dataset/fiber-data_256x256 --dir /lustre/ssd/ws/dchristi-diffusion2/checkpoints/fiber_1x256x256_test_w_descriptor"


python scripts/image_train.py $DIR_TRAIN $MODEL_FLAGS $CHECK_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS



exit 0
