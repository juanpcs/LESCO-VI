import tensorflow as tf

# List all available devices
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs disponibles:", physical_devices)
