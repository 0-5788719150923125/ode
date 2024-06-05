# Find CUDA libs
find / -name cuda

# Export path to CUDA libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/crow/Repos/ode/cuda/lib

# Set GPU device index
export CUDA_VISIBLE_DEVICES=0