#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define weight arrays (adjust sizes based on model)
#define CONV1_FILTERS 6
#define CONV1_KERNEL_SIZE 5
#define FC1_SIZE 120
#define FC2_SIZE 84
#define FC3_SIZE 10

float conv1_weights[CONV1_FILTERS][CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE];
float conv1_biases[CONV1_FILTERS];
float fc1_weights[FC1_SIZE][400]; // 400 = 16 * 4 * 4 (flattened conv2 output)
float fc1_biases[FC1_SIZE];
float fc2_weights[FC2_SIZE][FC1_SIZE];
float fc2_biases[FC2_SIZE];
float fc3_weights[FC3_SIZE][FC2_SIZE];
float fc3_biases[FC3_SIZE];

// Function to load weights from file
void load_weights(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    char layer_name[50];  // To store layer names
    int num_values;       // Number of values to read
    float value;          // Temporary variable to hold a value

    // Read Conv1 Weights
    fscanf(file, "%s %d", layer_name, &num_values); // Read "conv1.weight" and count
    for (int i = 0; i < CONV1_FILTERS; i++) {
        for (int j = 0; j < CONV1_KERNEL_SIZE; j++) {
            for (int k = 0; k < CONV1_KERNEL_SIZE; k++) {
                fscanf(file, "%f", &conv1_weights[i][j][k]);
            }
        }
    }

    // Read Conv1 Biases
    fscanf(file, "%s %d", layer_name, &num_values); // Read "conv1.bias" and count
    for (int i = 0; i < CONV1_FILTERS; i++) {
        fscanf(file, "%f", &conv1_biases[i]);
    }

    // Read FC1 Weights
    fscanf(file, "%s %d", layer_name, &num_values); // Read "fc1.weight" and count
    for (int i = 0; i < FC1_SIZE; i++) {
        for (int j = 0; j < 400; j++) {
            fscanf(file, "%f", &fc1_weights[i][j]);
        }
    }

    // Read FC1 Biases
    fscanf(file, "%s %d", layer_name, &num_values); // Read "fc1.bias" and count
    for (int i = 0; i < FC1_SIZE; i++) {
        fscanf(file, "%f", &fc1_biases[i]);
    }

    // Read FC2 Weights
    fscanf(file, "%s %d", layer_name, &num_values); // Read "fc2.weight" and count
    for (int i = 0; i < FC2_SIZE; i++) {
        for (int j = 0; j < FC1_SIZE; j++) {
            fscanf(file, "%f", &fc2_weights[i][j]);
        }
    }

    // Read FC2 Biases
    fscanf(file, "%s %d", layer_name, &num_values); // Read "fc2.bias" and count
    for (int i = 0; i < FC2_SIZE; i++) {
        fscanf(file, "%f", &fc2_biases[i]);
    }

    // Read FC3 Weights
    fscanf(file, "%s %d", layer_name, &num_values); // Read "fc3.weight" and count
    for (int i = 0; i < FC3_SIZE; i++) {
        for (int j = 0; j < FC2_SIZE; j++) {
            fscanf(file, "%f", &fc3_weights[i][j]);
        }
    }

    // Read FC3 Biases
    fscanf(file, "%s %d", layer_name, &num_values); // Read "fc3.bias" and count
    for (int i = 0; i < FC3_SIZE; i++) {
        fscanf(file, "%f", &fc3_biases[i]);
    }

    fclose(file);
    printf("Weights loaded successfully!\n");
}

// Test the function
int main() {
    load_weights("weights.txt");

    // Print some loaded values for verification
    printf("Conv1 first filter:\n");
    for (int i = 0; i < CONV1_KERNEL_SIZE; i++) {
        for (int j = 0; j < CONV1_KERNEL_SIZE; j++) {
            printf("%f ", conv1_weights[0][i][j]);
        }
        printf("\n");
    }

    return 0;
}
