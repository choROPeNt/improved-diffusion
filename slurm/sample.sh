#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --account=p_autoshear
#SBATCH --partition=alpha-interactive
#SBATCH --gres=gpu:1
#SBATCH --job-name=DIFFUSION_Sample_%j
#SBATCH --output=diffusion_sample_test_%j.out    
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=paul.seibert@tu-dresden.de

module load modenv/hiera  GCC/10.2.0  CUDA/11.7.0
module load Python/3.8.6
module load OpenMPI/4.0.5

source /lustre/ssd/ws/dchristi-diffusion2/.venv_cpu/bin/activate
## MODEL PARAMETERS
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma FALSE --num_classes=3 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 3000 --noise_schedule linear"

## specify model checkpoint folder
MODELDIR="/lustre/ssd/ws/dchristi-diffusion2/checkpoints/fiber_1x256x256_test_w_descriptor" 
# MODELLIST="
# model000000.pt 
# model010000.pt 
# model020000.pt 
# model030000.pt 
# model040000.pt 
# model050000.pt 
# model060000.pt 
# model070000.pt 
# model080000.pt 
# model090000.pt 
# model100000.pt"

# MODELLIST="
# model110000.pt
# model120000.pt
# model130000.pt
# model140000.pt
# model150000.pt
# model160000.pt
# model170000.pt
# model180000.pt
# model190000.pt
# # model200000.pt"

# MODELLIST="
# model210000.pt
# model220000.pt
# model230000.pt
# model240000.pt
# model250000.pt
# model260000.pt
# model270000.pt
# model280000.pt
# model290000.pt
# model300000.pt"

# MODELLIST="
# model080000.pt
# model090000.pt
# model100000.pt
# model190000.pt
# model200000.pt
# model290000.pt
# model300000.pt"

MODELLIST="
model073000.pt
model154000.pt"

## LOOP over specified $MODELLIST
for MODEL in ${MODELLIST};do echo ${MODELDIR}/${MODEL};
    echo sampling ${MODEL} from ${MODELDIR}
    SAMPLE="--model_path ${MODELDIR}/${MODEL} --dir /lustre/ssd/ws/dchristi-diffusion2/samples/fiber_1x256x256_test_w_descriptor/${MODEL}"; 
    echo ${SAMPLE};
    python scripts/image_sample_class.py $SAMPLE $MODEL_FLAGS $DIFFUSION_FLAGS 
    echo "===================================================================================" 
done;



exit 0
