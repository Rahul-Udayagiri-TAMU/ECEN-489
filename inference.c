#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG_SIZE 28

// Define weight arrays (adjust sizes based on model)
#define CONV1_FILTERS 6
#define CONV1_KERNEL_SIZE 5
#define CONV2_FILTERS 16
#define CONV2_KERNEL_SIZE 5
#define FC1_SIZE 120
#define FC2_SIZE 84
#define FC3_SIZE 10

float input_image[IMG_SIZE][IMG_SIZE];

float conv1_weights[CONV1_FILTERS][CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE];
float conv1_biases[CONV1_FILTERS];
float conv2_weights[CONV2_FILTERS][6][CONV2_KERNEL_SIZE][CONV2_KERNEL_SIZE];
float conv2_biases[CONV2_FILTERS];
float fc1_weights[FC1_SIZE][256]; // 256 = 16 * 4 * 4
float fc1_biases[FC1_SIZE];
float fc2_weights[FC2_SIZE][FC1_SIZE];
float fc2_biases[FC2_SIZE];
float fc3_weights[FC3_SIZE][FC2_SIZE];
float fc3_biases[FC3_SIZE];



#define CONV1_OUTPUT_SIZE 24
#define POOL1_SIZE 2
#define POOL1_OUTPUT_SIZE 12

float input_image[IMG_SIZE][IMG_SIZE];                      // normalized input image
float conv1_output[CONV1_FILTERS][CONV1_OUTPUT_SIZE][CONV1_OUTPUT_SIZE]; // after conv1
float pool1_output[CONV1_FILTERS][POOL1_OUTPUT_SIZE][POOL1_OUTPUT_SIZE]; // after pool1

// Perform convolution (valid, stride 1)
void conv1_forward() {
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 0; i < CONV1_OUTPUT_SIZE; i++) {
            for (int j = 0; j < CONV1_OUTPUT_SIZE; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < CONV1_KERNEL_SIZE; ki++) {
                    for (int kj = 0; kj < CONV1_KERNEL_SIZE; kj++) {
                        sum += input_image[i + ki][j + kj] * conv1_weights[f][ki][kj];
                    }
                }
                conv1_output[f][i][j] = fmaxf(0.0f, sum + conv1_biases[f]);  // ReLU activation
            }
        }
    }
}

// Max pooling with 2x2 kernel and stride 2
void pool1_forward() {
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 0; i < POOL1_OUTPUT_SIZE; i++) {
            for (int j = 0; j < POOL1_OUTPUT_SIZE; j++) {
                float max_val = -1e9;
                for (int ki = 0; ki < POOL1_SIZE; ki++) {
                    for (int kj = 0; kj < POOL1_SIZE; kj++) {
                        int row = i * POOL1_SIZE + ki;
                        int col = j * POOL1_SIZE + kj;
                        if (conv1_output[f][row][col] > max_val) {
                            max_val = conv1_output[f][row][col];
                        }
                    }
                }
                pool1_output[f][i][j] = max_val;
            }
        }
    }
}




#define CONV2_INPUT_SIZE 12
#define CONV2_OUTPUT_SIZE 8
#define POOL2_OUTPUT_SIZE 4

float conv2_output[CONV2_FILTERS][CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE];
float pool2_output[CONV2_FILTERS][POOL2_OUTPUT_SIZE][POOL2_OUTPUT_SIZE];


void conv2_forward() {
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < CONV2_OUTPUT_SIZE; i++) {
            for (int j = 0; j < CONV2_OUTPUT_SIZE; j++) {
                float sum = 0.0f;
                for (int c = 0; c < CONV1_FILTERS; c++) {
                    for (int ki = 0; ki < CONV2_KERNEL_SIZE; ki++) {
                        for (int kj = 0; kj < CONV2_KERNEL_SIZE; kj++) {
                            sum += pool1_output[c][i + ki][j + kj] * conv2_weights[f][c][ki][kj];
                        }
                    }
                }
                conv2_output[f][i][j] = fmaxf(0.0f, sum + conv2_biases[f]);  // ReLU
            }
        }
    }
}


void pool2_forward() {
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < POOL2_OUTPUT_SIZE; i++) {
            for (int j = 0; j < POOL2_OUTPUT_SIZE; j++) {
                float max_val = -1e9;
                for (int ki = 0; ki < POOL1_SIZE; ki++) {
                    for (int kj = 0; kj < POOL1_SIZE; kj++) {
                        int row = i * POOL1_SIZE + ki;
                        int col = j * POOL1_SIZE + kj;
                        if (conv2_output[f][row][col] > max_val) {
                            max_val = conv2_output[f][row][col];
                        }
                    }
                }
                pool2_output[f][i][j] = max_val;
    //            printf("%f ", pool2_output[f][i][j]);
            }
//            printf("\n");
        }
  //      printf("\n");
    }
}



float fc_input[256];  // Flattened input for FC1
float fc1_output[FC1_SIZE];  // 120
float fc2_output[FC2_SIZE];  // 84
float fc3_output[FC3_SIZE];  // 10 (final scores)

void flatten_pool2_output() {
    int idx = 0;
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < POOL2_OUTPUT_SIZE; i++) {
            for (int j = 0; j < POOL2_OUTPUT_SIZE; j++) {
                fc_input[idx++] = pool2_output[f][i][j];
            }
        }
    }
}

void fc1_forward() {
    for (int i = 0; i < FC1_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 256; j++) {
            sum += fc_input[j] * fc1_weights[i][j];
        }
        fc1_output[i] = fmaxf(0.0f, sum + fc1_biases[i]);  // ReLU
    }
}


void fc2_forward() {
    for (int i = 0; i < FC2_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < FC1_SIZE; j++) {
            sum += fc1_output[j] * fc2_weights[i][j];
        }
        fc2_output[i] = fmaxf(0.0f, sum + fc2_biases[i]);  // ReLU
    }
}

void fc3_forward() {
    for (int i = 0; i < FC3_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < FC2_SIZE; j++) {
            sum += fc2_output[j] * fc3_weights[i][j];
        }
        fc3_output[i] = sum + fc3_biases[i];  // No ReLU here
    }
}

int predict_label() {
    int max_index = 0;
    float max_val = fc3_output[0];
    for (int i = 1; i < FC3_SIZE; i++) {
        if (fc3_output[i] > max_val) {
            max_val = fc3_output[i];
            max_index = i;
        }
    }
    return max_index;
}

void softmax(float* input, float* output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; ++i) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    float sum = 0.0;
    for (int i = 0; i < length; ++i) {
        output[i] = expf(input[i] - max_val); // for numerical stability
        sum += output[i];
    }

    for (int i = 0; i < length; ++i) {
        output[i] /= sum;
    }
}

// Function to load weights from file
void load_weights(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    char layer_name[50];
    int num_values;

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < CONV1_FILTERS; i++)
        for (int j = 0; j < CONV1_KERNEL_SIZE; j++)
            for (int k = 0; k < CONV1_KERNEL_SIZE; k++)
                fscanf(file, "%f", &conv1_weights[i][j][k]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < CONV1_FILTERS; i++)
        fscanf(file, "%f", &conv1_biases[i]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int l = 0; l < CONV2_FILTERS; l++)
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < CONV2_KERNEL_SIZE; j++)
                for (int k = 0; k < CONV2_KERNEL_SIZE; k++)
                    fscanf(file, "%f", &conv2_weights[l][i][j][k]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < CONV2_FILTERS; i++)
        fscanf(file, "%f", &conv2_biases[i]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < FC1_SIZE; i++)
        for (int j = 0; j < 256; j++)
            fscanf(file, "%f", &fc1_weights[i][j]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < FC1_SIZE; i++)
        fscanf(file, "%f", &fc1_biases[i]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < FC2_SIZE; i++)
        for (int j = 0; j < FC1_SIZE; j++)
            fscanf(file, "%f", &fc2_weights[i][j]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < FC2_SIZE; i++)
        fscanf(file, "%f", &fc2_biases[i]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < FC3_SIZE; i++)
        for (int j = 0; j < FC2_SIZE; j++)
            fscanf(file, "%f", &fc3_weights[i][j]);

    fscanf(file, "%s %d", layer_name, &num_values);
    for (int i = 0; i < FC3_SIZE; i++)
        fscanf(file, "%f", &fc3_biases[i]);

    fclose(file);
    printf("Weights loaded successfully!\n");
}

// Load and normalize a grayscale MNIST-style image
void load_input_image(const char* filename) {
    int width, height, channels;
    unsigned char* img_data = stbi_load(filename, &width, &height, &channels, 1); // force grayscale

    if (!img_data) {
        printf("Failed to load image: %s\n", filename);
        exit(1);
    }

    if (width != IMG_SIZE || height != IMG_SIZE) {
        printf("Error: Image must be 28x28. Got %dx%d\n", width, height);
        stbi_image_free(img_data);
        exit(1);
    }

    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            input_image[i][j] = img_data[i * IMG_SIZE + j] / 255.0f;
        }
    }

    stbi_image_free(img_data);
    printf("Input image loaded and normalized successfully.\n");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <28x28_grayscale_image.png>\n", argv[0]);
        return 1;
    }

    load_weights("weights.txt");
    load_input_image(argv[1]);

    conv1_forward();
    pool1_forward();
    conv2_forward();
    pool2_forward();
    conv1_forward();
    pool1_forward();
    conv2_forward();
    pool2_forward();
    flatten_pool2_output();
    fc1_forward();
    fc2_forward();
    fc3_forward();

    int prediction = predict_label();
    printf("Predicted Digit: %d\n", prediction);
   
    float probabilities[10];
    softmax(fc3_output, probabilities, 10);
  
   for (int i = 0; i < 10; ++i) {
        printf("Class %d: %.4f\n", i, probabilities[i]);
   } 
    return 0;
}
