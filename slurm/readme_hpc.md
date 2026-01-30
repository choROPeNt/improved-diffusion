srun \
  -p capella \
  -N 1 \
  -n 1 \
  --gres=gpu:1 \
  --gpus-per-task=1 \
  -c 12 \
  --mem-per-cpu=12G \
  -t 04:00:00 \
  --account=p_biiax \
  --pty /bin/bash -l

ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0 OpenMPI/5.0.3


source /data/horse/ws/dchristi-diffusion/.venv/bin/activate

export OMPI_MCA_smsc=^knem

python -c "import torch;print(torch.__version__);print(torch.cuda.is_available());print(torch.cuda.get_device_name())"