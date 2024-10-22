import numpy as np
import os

def convert_to_c_array(npy_file, output_file):
    # Load the numpy array from the file
    data = np.load(npy_file)
    
    # Open the output file and write the array in C format
    with open(output_file, 'w') as f:
        # Get a C-compatible name for the array from the file name
        array_name = npy_file.replace('.npy', '').replace('/', '_').replace('\\', '_')
        f.write(f'float {array_name}[] = {{\n')
        
        # Write each value in the array with 6 decimal places
        for i, value in enumerate(data.flatten()):
            if i % 8 == 0:
                f.write('\n    ')
            f.write(f'{value:.6f}, ')
        f.write('\n};\n')

# List of weight and bias files based on your file names
weight_files = [
    'weights_layer_0_weights.npy', 'weights_layer_0_biases.npy',
    'weights_layer_1_weights.npy', 'weights_layer_1_biases.npy',
    'weights_layer_4_weights.npy', 'weights_layer_4_biases.npy',
    'weights_layer_5_weights.npy', 'weights_layer_5_biases.npy',
    'weights_layer_9_weights.npy', 'weights_layer_9_biases.npy',
    'weights_layer_11_weights.npy', 'weights_layer_11_biases.npy'
]

# Convert each file in the list
for npy_file in weight_files:
    # Generate the output filename by replacing .npy with .c
    output_file = npy_file.replace('.npy', '.c')
    convert_to_c_array(npy_file, output_file)
