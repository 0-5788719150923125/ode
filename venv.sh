# Setup a virtual environment for Node.js
python3 -m venv ./env
chmod +x env/bin/activate
. env/bin/activate

# Install Tensorflow for Python (required by Tensorflow.js)
python3 -m pip install tensorflow[and-cuda]

# Export path to CUDA libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/crow/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/crow/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/data/documents/AI-Horde-Worker-main/conda/pkgs/cudatoolkit-11.8.0-h4ba93d1_12/lib

# Find CUDA libs
find / -name cuda_runtime

# Set GPU device index
export CUDA_VISIBLE_DEVICES=0

# GPU test
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"