#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define INPUT_SIZE 28
#define FILTER_SIZE 5
#define OUTPUT_SIZE 24
#define NUM_FILTERS 6


#define CONV2_IN_CHANNELS 6
#define CONV2_OUT_CHANNELS 16
#define CONV2_FILTER_SIZE 5
#define CONV2_OUT_SIZE 8
#define CONV2_NUM_WEIGHTS (CONV2_OUT_CHANNELS * CONV2_IN_CHANNELS * CONV2_FILTER_SIZE * CONV2_FILTER_SIZE)  // 2400



#define cudaCheckError() {\
	cudaError_t e = cudaGetLastError();\
	\
	if(e != cudaSuccess) {\
		printf("CUDA Failure %s, %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));\
	}\
}\


__global__ void softmax_kernel(const float* input, float* output, int length) {
    __shared__ float max_val;
    __shared__ float sum;

    // Step 1: Find max value for numerical stability (thread 0)
    if (threadIdx.x == 0) {
        max_val = input[0];
        for (int i = 1; i < length; ++i) {
            if (input[i] > max_val) max_val = input[i];
        }
    }
    __syncthreads();

    // Step 2: Compute exponentials and accumulate sum
    if (threadIdx.x < length) {
        output[threadIdx.x] = expf(input[threadIdx.x] - max_val);
    }
    __syncthreads();

    // Step 3: Compute sum of exponentials (thread 0)
    if (threadIdx.x == 0) {
        sum = 0.0f;
        for (int i = 0; i < length; ++i) {
            sum += output[i];
        }
    }
    __syncthreads();

    // Step 4: Normalize to get probabilities
    if (threadIdx.x < length) {
        output[threadIdx.x] /= sum;
    }
}

__global__ void fc3_kernel(
    const float* input,   // [84]
    const float* weights, // [10 x 84]
    const float* biases,  // [10]
    float* output         // [10]
) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= 10) return;

    float sum = 0.0f;
    for (int i = 0; i < 84; ++i) {
        sum += input[i] * weights[neuron_id * 84 + i];
    }

    output[neuron_id] = sum + biases[neuron_id]; // No ReLU here
}

__global__ void fc2_kernel(
    const float* input,   // [120]
    const float* weights, // [84 x 120]
    const float* biases,  // [84]
    float* output         // [84]
) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= 84) return;

    float sum = 0.0f;
    for (int i = 0; i < 120; ++i) {
        sum += input[i] * weights[neuron_id * 120 + i];
    }

    output[neuron_id] = fmaxf(0.0f, sum + biases[neuron_id]); // ReLU
}

__global__ void fc1_kernel(
    const float* input,   // [256]
    const float* weights, // [120 x 256]
    const float* biases,  // [120]
    float* output         // [120]
) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= 120) return;

    float sum = 0.0f;
    for (int i = 0; i < 256; ++i) {
        sum += input[i] * weights[neuron_id * 256 + i];
    }

    output[neuron_id] = fmaxf(0.0f, sum + biases[neuron_id]); // ReLU
}


__global__ void flatten_pool2(
    const float* input,  // [16 x 4 x 4]
    float* output        // [256]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 256) return;

    output[idx] = input[idx]; // data is already in correct layout
}

__global__ void pool2_shared_kernel(
    const float* input,   // [16 x 8 x 8]
    float* output         // [16 x 4 x 4]
) {
    const int fmap = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_row = blockIdx.y * blockDim.y + ty;
    const int out_col = blockIdx.x * blockDim.x + tx;

    if (out_row >= 4 || out_col >= 4) return;

    // Shared memory tile for 2x2 region (+1 to prevent bank conflict or overlap)
    __shared__ float tile[10][10];  // enough for 2x2 pooling on 8x8 with 2x2 stride

    // Load the required 2x2 region into shared memory
    int in_row = out_row * 2;
    int in_col = out_col * 2;

    // Each thread loads its 2x2 patch into shared memory
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int r = in_row + i;
            int c = in_col + j;
            if (r < 8 && c < 8) {
                tile[ty * 2 + i][tx * 2 + j] = input[fmap * 64 + r * 8 + c];
            }
        }
    }

    __syncthreads();

    // Do pooling using shared memory
    float max_val = -1e9;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            max_val = fmaxf(max_val, tile[ty * 2 + i][tx * 2 + j]);
        }
    }

    int out_idx = fmap * 16 + out_row * 4 + out_col;
    output[out_idx] = max_val;
}

__global__ void conv2_shared_kernel(
    const float* input,       // [6 x 12 x 12]
    const float* filters,     // [16 x 6 x 5 x 5] = 2400
    const float* biases,      // [16]
    float* output             // [16 x 8 x 8]
) {
    const int out_ch = blockIdx.z;                     // output channel (0–15)
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= 8 || col >= 8) return;

    // Shared memory tile: enough for 8x8 output + 4 halo → 12x12 region
    __shared__ float tile[6][12 + 4][12 + 4]; // 6 input channels, 16x16 shared

    // Load shared memory for each input channel
    for (int c = 0; c < 6; ++c) {
        int global_row = row + 0;
        int global_col = col + 0;

        for (int i = threadIdx.y; i < 12 + 4; i += blockDim.y) {
            for (int j = threadIdx.x; j < 12 + 4; j += blockDim.x) {
                int src_row = blockIdx.y * blockDim.y + i;
                int src_col = blockIdx.x * blockDim.x + j;

                float val = 0.0f;
                if (src_row < 12 && src_col < 12)
                    val = input[c * 144 + src_row * 12 + src_col];

                tile[c][i][j] = val;
            }
        }
    }

    __syncthreads();

    float sum = 0.0f;

    for (int c = 0; c < 6; ++c) {
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                int t_row = threadIdx.y + i;
                int t_col = threadIdx.x + j;

                float val = tile[c][t_row][t_col];
                int f_idx = out_ch * (6 * 25) + c * 25 + i * 5 + j;
                sum += val * filters[f_idx];
            }
        }
    }

    int out_idx = out_ch * 64 + row * 8 + col;
    output[out_idx] = fmaxf(0.0f, sum + biases[out_ch]);  // ReLU
}



__global__ void conv1_shared_kernel(
    const float* input,       // [28 x 28]
    const float* filters,     // [6 x 5 x 5]
    const float* biases,      // [6]
    float* output             // [6 x 24 x 24]
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    const int filter_id = blockIdx.z;  // 0–5

    // Shared memory tile: enough for 24x24 output + 4 halo (5x5 filter needs 4 extra rows/cols)
    __shared__ float tile[28 + 4][28 + 4];  // 32x32

    // Load tile from input into shared memory (with zero-padding)
    int shared_row = ty;
    int shared_col = tx;
    int input_row = blockIdx.y * blockDim.y + shared_row;
    int input_col = blockIdx.x * blockDim.x + shared_col;

    if (input_row < 28 && input_col < 28)
        tile[shared_row][shared_col] = input[input_row * 28 + input_col];
    else
        tile[shared_row][shared_col] = 0.0f;

    __syncthreads();

    // Only compute output if we're in valid 24x24 output space
    if (row < 24 && col < 24) {
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                int s_row = ty + i;
                int s_col = tx + j;
                if (s_row < 32 && s_col < 32) {
                    int filter_idx = filter_id * 25 + i * 5 + j;
                    sum += tile[s_row][s_col] * filters[filter_idx];
                }
            }
        }

        int out_idx = filter_id * 24 * 24 + row * 24 + col;
        output[out_idx] = fmaxf(0.0f, sum + biases[filter_id]); // ReLU
    }
}


__global__ void pool1_shared_kernel(
    const float* input,    // [6 × 24 × 24]
    float* output          // [6 × 12 × 12]
) {
    int fmap = blockIdx.z; // Feature map index (0–5)
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row >= 12 || out_col >= 12) return;

    // Shared memory tile for 2×2 pooling input
    __shared__ float tile[2][2];

    // Each thread loads its own 2x2 region (non-overlapping)
    int in_row = out_row * 2;
    int in_col = out_col * 2;

    float max_val = -1e9;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int idx = fmap * 24 * 24 + (in_row + i) * 24 + (in_col + j);
            float val = input[idx];
            if (val > max_val) max_val = val;
        }
    }

    int out_idx = fmap * 12 * 12 + out_row * 12 + out_col;
    output[out_idx] = max_val;
}



// Helper function to read flattened input (28x28 = 784)
void read_input(const char* filename, float* input) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Cannot open %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < 28 * 28; i++) {
        if (fscanf(f, "%f", &input[i]) != 1) {
            printf("Error: Failed to read float %d from %s\n", i, filename);
            fclose(f);
            exit(1);
        }
    }

    fclose(f);
}

// Read filters: 6 filters of size 5x5 (total 6x25)
void read_filters(const char* filename, float* filters) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Cannot open %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < NUM_FILTERS * 25; i++) {
        if (fscanf(f, "%f", &filters[i]) != 1) {
            printf("Error: Failed to read float %d from %s\n", i, filename);
            fclose(f);
            exit(1);
        }
    }

    fclose(f);
}

// Read biases: 6 biases
void read_biases(const char* filename, float* biases) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Cannot open %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < NUM_FILTERS; i++) {
        if (fscanf(f, "%f", &biases[i]) != 1) {
            printf("Error: Failed to read float %d from %s\n", i, filename);
            fclose(f);
            exit(1);
        }
    }

    fclose(f);
}



void read_conv2_filters(const char* filename, float* filters) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < CONV2_NUM_WEIGHTS; i++) {
        fscanf(f, "%f", &filters[i]);
    }
    fclose(f);
}

void read_conv2_biases(const char* filename, float* biases) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < CONV2_OUT_CHANNELS; i++) {
        fscanf(f, "%f", &biases[i]);
    }
    fclose(f);
}




void read_fc1_weights(const char* filename, float* weights) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < 120 * 256; i++) {
        fscanf(f, "%f", &weights[i]);
    }
    fclose(f);
}

void read_fc1_biases(const char* filename, float* biases) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < 120; i++) {
        fscanf(f, "%f", &biases[i]);
    }
    fclose(f);
}


void read_fc2_weights(const char* filename, float* weights) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < 84 * 120; i++) {
        fscanf(f, "%f", &weights[i]);
    }
    fclose(f);
}

void read_fc2_biases(const char* filename, float* biases) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < 84; i++) {
        fscanf(f, "%f", &biases[i]);
    }
    fclose(f);
}


void read_fc3_weights(const char* filename, float* weights) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < 10 * 84; i++) {
        fscanf(f, "%f", &weights[i]);
    }
    fclose(f);
}

void read_fc3_biases(const char* filename, float* biases) {
    FILE* f = fopen(filename, "r");
    for (int i = 0; i < 10; i++) {
        fscanf(f, "%f", &biases[i]);
    }
    fclose(f);
}

// Save output
void save_output(const char* filename, float* output) {
    FILE* f = fopen(filename, "w");
    for (int f_id = 0; f_id < NUM_FILTERS; f_id++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                int idx = f_id * OUTPUT_SIZE * OUTPUT_SIZE + i * OUTPUT_SIZE + j;
                fprintf(f, "%.6f ", output[idx]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void save_pool1_output(const char* filename, float* output) {
	FILE* f = fopen(filename, "w");
	for (int f_idx = 0; f_idx < 6; f_idx++) {
	    for (int i = 0; i < 12; i++) {
		for (int j = 0; j < 12; j++) {
		    int idx = f_idx * 144 + i * 12 + j;
		    fprintf(f, "%.6f ", output[idx]);
		}
		fprintf(f, "\n");
	    }
	    fprintf(f, "\n");
	}
	fclose(f);
}

void save_conv2_output(const char* filename, float* output) {
	FILE* f = fopen(filename, "w");
	for (int f_id = 0; f_id < 16; f_id++) {
	    for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
		    int idx = f_id * 64 + i * 8 + j;
		    fprintf(f, "%.6f ", output[idx]);
		}
		fprintf(f, "\n");
	    }
	    fprintf(f, "\n");
	}
	fclose(f);
}

void save_pool2_output(const char* filename, float* output) {
	FILE* f = fopen(filename, "w");
	for (int f_id = 0; f_id < 16; f_id++) {
	    for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
		    int idx = f_id * 16 + i * 4 + j;
		    fprintf(f, "%.6f ", output[idx]);
		}
		fprintf(f, "\n");
	    }
	    fprintf(f, "\n");
	}
	fclose(f);
}

void save_fc1_output(const char* filename, float* output) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < 120; ++i) {
	    fprintf(f, "%.6f\n", output[i]);
	}
	fclose(f);
}

void save_fc2_output(const char* filename, float* output) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < 84; i++) {
		fprintf(f, "%.6f\n", output[i]);
	}
	fclose(f);
}


void save_fc3_output(const char* filename, float* output) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < 10; i++) {
		fprintf(f, "%.6f\n", output[i]);
	}
	fclose(f);
}

int main() {
    float *h_input, *h_filters, *h_biases, *h_output;

    // Allocate host memory
    h_input = (float*)malloc(28 * 28 * sizeof(float));
    h_filters = (float*)malloc(NUM_FILTERS * 25 * sizeof(float));
    h_biases = (float*)malloc(NUM_FILTERS * sizeof(float));
    h_output = (float*)malloc(NUM_FILTERS * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));

    // Read from files with checks
    printf("[INFO] Reading conv1 input and weights...\n");
    read_input("OutputFiles/input_image.txt", h_input);
    read_filters("OutputFiles/conv1_weights.txt", h_filters);
    read_biases("OutputFiles/conv1_biases.txt", h_biases);


    // Allocate device memory
    float *d_input, *d_filters, *d_biases, *d_output;
    cudaMalloc(&d_input, 28 * 28 * sizeof(float));
    cudaMalloc(&d_filters, NUM_FILTERS * 25 * sizeof(float));
    cudaMalloc(&d_biases, NUM_FILTERS * sizeof(float));
    cudaMalloc(&d_output, NUM_FILTERS * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaCheckError();

    // Copy data to device
    cudaMemcpy(d_input, h_input, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filters, h_filters, NUM_FILTERS * 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases, NUM_FILTERS * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Launch conv1
    printf("[INFO] Launching conv1_shared_kernel...\n");
    dim3 blockDim(12, 12);  // safer with shared memory
    dim3 gridDim((OUTPUT_SIZE + 11) / 12, (OUTPUT_SIZE + 11) / 12, NUM_FILTERS);
    conv1_shared_kernel<<<gridDim, blockDim>>>(d_input, d_filters, d_biases, d_output);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("[INFO] conv1 done.\n");

    // Pool1
    float *d_pool1_output;
    cudaMalloc(&d_pool1_output, 6 * 12 * 12 * sizeof(float));
    cudaCheckError();

    dim3 blockDimPool1(12, 12);
    dim3 gridDimPool1((12 + 11) / 12, (12 + 11) / 12, 6);

    printf("[INFO] Launching pool1_kernel...\n");
    pool1_shared_kernel<<<gridDimPool1, blockDimPool1>>>(d_output, d_pool1_output);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("[INFO] pool1 done.\n");

    // Conv2
    printf("[INFO] Reading conv2 weights...\n");
    float* h_conv2_filters = (float*)malloc(2400 * sizeof(float)); // 16x6x25
    float* h_conv2_biases = (float*)malloc(16 * sizeof(float));
    float* h_conv2_output = (float*)malloc(16 * 8 * 8 * sizeof(float));
    read_conv2_filters("OutputFiles/conv2_weights.txt", h_conv2_filters);
    read_conv2_biases("OutputFiles/conv2_biases.txt", h_conv2_biases);

    float *d_conv2_filters, *d_conv2_biases, *d_conv2_output;
    cudaMalloc(&d_conv2_filters, 2400 * sizeof(float));
    cudaMalloc(&d_conv2_biases, 16 * sizeof(float));
    cudaMalloc(&d_conv2_output, 16 * 8 * 8 * sizeof(float));
    cudaCheckError();

    cudaMemcpy(d_conv2_filters, h_conv2_filters, 2400 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_biases, h_conv2_biases, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 blockDimConv2(8, 8);
    dim3 gridDimConv2((8 + 7) / 8, (8 + 7) / 8, 16);

    printf("[INFO] Launching conv2_kernel...\n");
    conv2_shared_kernel<<<gridDimConv2, blockDimConv2>>>(d_pool1_output, d_conv2_filters, d_conv2_biases, d_conv2_output);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("[INFO] conv2 done.\n");

    // Pool2
    float* h_pool2_output = (float*)malloc(16 * 4 * 4 * sizeof(float));
    float* d_pool2_output;
    cudaMalloc(&d_pool2_output, 16 * 4 * 4 * sizeof(float));
    cudaCheckError();

    dim3 blockDimPool2(4, 4);
    dim3 gridDimPool2((4 + 3) / 4, (4 + 3) / 4, 16);

    printf("[INFO] Launching pool2_kernel...\n");
    pool2_shared_kernel<<<gridDimPool2, blockDimPool2>>>(d_conv2_output, d_pool2_output);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("[INFO] pool2 done.\n");

    // FC1
    printf("[INFO] Reading fc1 weights...\n");
    float* h_fc1_weights = (float*)malloc(120 * 256 * sizeof(float));
    float* h_fc1_biases = (float*)malloc(120 * sizeof(float));
    float* h_fc1_output = (float*)malloc(120 * sizeof(float));
    read_fc1_weights("OutputFiles/fc1_weights.txt", h_fc1_weights);
    read_fc1_biases("OutputFiles/fc1_biases.txt", h_fc1_biases);


    float *d_fc1_weights, *d_fc1_biases, *d_fc1_output, *d_fc1_input;
    cudaMalloc(&d_fc1_weights, 120 * 256 * sizeof(float));
    cudaMalloc(&d_fc1_biases, 120 * sizeof(float));
    cudaMalloc(&d_fc1_output, 120 * sizeof(float));
    cudaMalloc(&d_fc1_input, 256 * sizeof(float));
    cudaCheckError();

    cudaMemcpy(d_fc1_weights, h_fc1_weights, 120 * 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_biases, h_fc1_biases, 120 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    flatten_pool2<<<1, 256>>>(d_pool2_output, d_fc1_input);
    cudaDeviceSynchronize();
    cudaCheckError();

    fc1_kernel<<<1, 120>>>(d_fc1_input, d_fc1_weights, d_fc1_biases, d_fc1_output);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("[INFO] fc1 done.\n");

    // FC2
    printf("[INFO] Reading fc2 weights...\n");
    float *h_fc2_weights = (float*)malloc(84 * 120 * sizeof(float));
    float *h_fc2_biases  = (float*)malloc(84 * sizeof(float));
    float *h_fc2_output  = (float*)malloc(84 * sizeof(float));
    read_fc2_weights("OutputFiles/fc2_weights.txt", h_fc2_weights);
    read_fc2_biases("OutputFiles/fc2_biases.txt", h_fc2_biases);

    float *d_fc2_weights, *d_fc2_biases, *d_fc2_output;
    cudaMalloc(&d_fc2_weights, 84 * 120 * sizeof(float));
    cudaMalloc(&d_fc2_biases,  84 * sizeof(float));
    cudaMalloc(&d_fc2_output,  84 * sizeof(float));
    cudaCheckError();

    cudaMemcpy(d_fc2_weights, h_fc2_weights, 84 * 120 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_biases,  h_fc2_biases,  84 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    fc2_kernel<<<1, 84>>>(d_fc1_output, d_fc2_weights, d_fc2_biases, d_fc2_output);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("[INFO] fc2 done.\n");

    // FC3
    printf("[INFO] Reading fc3 weights...\n");
    float *h_fc3_weights = (float*)malloc(10 * 84 * sizeof(float));
    float *h_fc3_biases  = (float*)malloc(10 * sizeof(float));
    float *h_fc3_output  = (float*)malloc(10 * sizeof(float));
    read_fc3_weights("OutputFiles/fc3_weights.txt", h_fc3_weights);
    read_fc3_biases("OutputFiles/fc3_biases.txt", h_fc3_biases);

    float *d_fc3_weights, *d_fc3_biases, *d_fc3_output;
    cudaMalloc(&d_fc3_weights, 10 * 84 * sizeof(float));
    cudaMalloc(&d_fc3_biases,  10 * sizeof(float));
    cudaMalloc(&d_fc3_output,  10 * sizeof(float));
    cudaCheckError();

    cudaMemcpy(d_fc3_weights, h_fc3_weights, 10 * 84 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_biases,  h_fc3_biases,  10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    fc3_kernel<<<1, 10>>>(d_fc2_output, d_fc3_weights, d_fc3_biases, d_fc3_output);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("[INFO] fc3 done.\n");

    // Softmax
    float* d_probabilities;
    float* h_probabilities = (float*)malloc(10 * sizeof(float));
    cudaMalloc(&d_probabilities, 10 * sizeof(float));
    cudaCheckError();

    softmax_kernel<<<1, 10>>>(d_fc3_output, d_probabilities, 10);
    cudaDeviceSynchronize();
    cudaCheckError();
    printf("[INFO] softmax done.\n");

    // Result
    cudaMemcpy(h_probabilities, d_probabilities, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    int predicted = 0;
    float max_prob = h_probabilities[0];
    for (int i = 1; i < 10; ++i) {
        if (h_probabilities[i] > max_prob) {
            max_prob = h_probabilities[i];
            predicted = i;
        }
    }

    printf("Predicted Digit: %d\n", predicted);
    for (int i = 0; i < 10; ++i) {
        printf("Class %d: %.4f\n", i, h_probabilities[i]);
    }

    // Final error check
    cudaDeviceSynchronize();
    cudaCheckError();
    return 0;
}
