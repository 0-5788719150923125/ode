# Find CUDA libs
# find / -name cuda

# Export path to CUDA libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/crow/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/crow/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/data/documents/AI-Horde-Worker-main/conda/pkgs/cudatoolkit-11.8.0-h4ba93d1_12/lib:/home/crow/repos/ode/cuda/lib

# Set GPU device index
export CUDA_VISIBLE_DEVICES=0