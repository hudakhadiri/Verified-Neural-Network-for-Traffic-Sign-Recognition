#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_HEIGHT 32    // Height of input images (change according to your input size)
#define INPUT_WIDTH 32     // Width of input images (change according to your input size)
#define NUM_CLASSES 43     // Number of classes (traffic signs)
#define LAYER_0_FILTERS 32 // Number of filters in layer 0
#define LAYER_1_FILTERS 64 // Number of filters in layer 1
#define LAYER_4_FILTERS 128 // Number of filters in layer 4
#define LAYER_5_FILTERS 256 // Number of filters in layer 5

// Include the converted weights and biases headers
#include "weights_layer_0_weights.c"
#include "weights_layer_0_biases.c"
#include "weights_layer_1_weights.c"
#include "weights_layer_1_biases.c"
#include "weights_layer_4_weights.c"
#include "weights_layer_4_biases.c"
#include "weights_layer_5_weights.c"
#include "weights_layer_5_biases.c"

// Function prototypes
void relu(float* input, int size);
void conv2d(float* input, float* output, float* weights, float* bias, int input_height, int input_width, int filter_height, int filter_width, int stride, int filters);
void maxpool2d(float* input, float* output, int input_height, int input_width, int pool_size);
void flatten(float* input, float* output, int height, int width);
void dense(float* input, float* weights, float* biases, float* output, int input_size, int output_size);
void softmax(float* input, float* output, int size);

// Main inference function
void cnn_inference(float* input_image) {
    // Layer 0
    float conv0_output[(INPUT_HEIGHT - 4) * (INPUT_WIDTH - 4) * LAYER_0_FILTERS]; // Output size depends on filter and stride
    conv2d(input_image, conv0_output, weights_layer_0_weights, weights_layer_0_biases, INPUT_HEIGHT, INPUT_WIDTH, 5, 5, 1, LAYER_0_FILTERS);
    relu(conv0_output, (INPUT_HEIGHT - 4) * (INPUT_WIDTH - 4) * LAYER_0_FILTERS);

    // Layer 1
    float conv1_output[(INPUT_HEIGHT - 8) * (INPUT_WIDTH - 8) * LAYER_1_FILTERS];
    conv2d(conv0_output, conv1_output, weights_layer_1_weights, weights_layer_1_biases, INPUT_HEIGHT - 4, INPUT_WIDTH - 4, 5, 5, 1, LAYER_1_FILTERS);
    relu(conv1_output, (INPUT_HEIGHT - 8) * (INPUT_WIDTH - 8) * LAYER_1_FILTERS);

    // Max pooling after layer 1
    float pool1_output[(INPUT_HEIGHT - 8) / 2 * (INPUT_WIDTH - 8) / 2 * LAYER_1_FILTERS];
    maxpool2d(conv1_output, pool1_output, INPUT_HEIGHT - 8, INPUT_WIDTH - 8, 2);

    // Flatten the output for the dense layer
    float flattened_output[(INPUT_HEIGHT - 8) / 2 * (INPUT_WIDTH - 8) / 2 * LAYER_1_FILTERS];
    flatten(pool1_output, flattened_output, (INPUT_HEIGHT - 8) / 2, (INPUT_WIDTH - 8) / 2 * LAYER_1_FILTERS);

    // Dense layer 0
    float dense0_output[512]; // Adjust the size according to the dense layer's output
    dense(flattened_output, weights_layer_4_weights, weights_layer_4_biases, dense0_output, (INPUT_HEIGHT - 8) / 2 * (INPUT_WIDTH - 8) / 2 * LAYER_1_FILTERS, 512);
    relu(dense0_output, 512);

    // Dense layer 1 (output layer)
    float output[NUM_CLASSES];
    dense(dense0_output, weights_layer_5_weights, weights_layer_5_biases, output, 512, NUM_CLASSES);
    
    // Apply softmax to get the probabilities
    float probabilities[NUM_CLASSES];
    softmax(output, probabilities, NUM_CLASSES);
    
    // Output the probabilities for each class
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("Class %d: %.6f\n", i, probabilities[i]);
    }
}

// Implementations of helper functions
void relu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = fmaxf(0, input[i]);
    }
}

void conv2d(float* input, float* output, float* weights, float* bias, int input_height, int input_width, int filter_height, int filter_width, int stride, int filters) {
    int out_height = (input_height - filter_height) / stride + 1;
    int out_width = (input_width - filter_width) / stride + 1;

    for (int f = 0; f < filters; f++) {
        for (int i = 0; i < out_height; i++) {
            for (int j = 0; j < out_width; j++) {
                float sum = 0;
                for (int ki = 0; ki < filter_height; ki++) {
                    for (int kj = 0; kj < filter_width; kj++) {
                        sum += input[(i * stride + ki) * input_width + (j * stride + kj)] * weights[f * filter_height * filter_width + ki * filter_width + kj];
                    }
                }
                output[i * out_width + j] = sum + bias[f];
            }
        }
    }
}

void maxpool2d(float* input, float* output, int input_height, int input_width, int pool_size) {
    int out_height = input_height / pool_size;
    int out_width = input_width / pool_size;

    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            float max_val = input[i * pool_size * input_width + j * pool_size];
            for (int ki = 0; ki < pool_size; ki++) {
                for (int kj = 0; kj < pool_size; kj++) {
                    max_val = fmaxf(max_val, input[(i * pool_size + ki) * input_width + (j * pool_size + kj)]);
                }
            }
            output[i * out_width + j] = max_val;
        }
    }
}

void flatten(float* input, float* output, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            output[i * width + j] = input[i * width + j];
        }
    }
}



void dense(float* input, float* weights, float* biases, float* output, int input_size, int output_size) {
    // Ensure output is initialized to zero
    for (int i = 0; i < output_size; i++) {
        output[i] = biases[i];  // Initialize with bias
    }

    // Calculate the output
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            // Ensure proper indexing to avoid segmentation faults
            if (i * input_size + j < (input_size * output_size)) {
                output[i] += input[j] * weights[i * input_size + j];
            }
        }
    }
}



void softmax(float* input, float* output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

int main() {
    float input_image[INPUT_HEIGHT * INPUT_WIDTH]; 

    // Perform inference
    cnn_inference(input_image);

    return 0;
}
