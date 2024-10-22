import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('./gtsrb.h5')

# Save each layer's weights
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if len(weights) > 0:
        np.save(f'weights_layer_{i}_weights.npy', weights[0])  # Save weights
        np.save(f'weights_layer_{i}_biases.npy', weights[1])   # Save biases
