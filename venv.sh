python3 -m venv ./env

chmod +x env/bin/activate

. env/bin/activate

python3 -m pip install tensorflow[and-cuda]

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
