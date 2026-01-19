import tensorflow as tf
import os

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Print GPU information
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:")
print(tf.config.list_physical_devices('GPU'))

# Test GPU with a simple operation
if tf.config.list_physical_devices('GPU'):
    print("\nTesting GPU with a simple operation:")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("GPU operation successful!")
else:
    print("\nNo GPU available for testing")