ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0 OpenMPI/5.0.3

source /data/horse/ws/dchristi-diffusion/.venv/bin/activate


python -c "import torch;print(torch.__version__);print(torch.cuda.is_available());print(torch.cuda.get_device_name())"