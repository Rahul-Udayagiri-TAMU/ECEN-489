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

//SOftmax is always implemented using Shared Memory
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

__global__ void pool2_kernel(
		const float* input,   // [16 * 8 * 8]
		float* output         // [16 * 4 * 4]
		) {
	int fmap = blockIdx.z;
	int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0–3
	int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0–3

	if (row >= 4 || col >= 4) return;

	int in_row = row * 2;
	int in_col = col * 2;

	float max_val = -1e9;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			int idx = fmap * 64 + (in_row + i) * 8 + (in_col + j); // 64 = 8x8
			float val = input[idx];
			if (val > max_val) max_val = val;
		}
	}

	int out_idx = fmap * 16 + row * 4 + col; // 16 = 4x4
	output[out_idx] = max_val;
}

__global__ void conv2_kernel(
		const float* input,       // [6 x 12 x 12]
		const float* filters,     // [16 x 6 x 5 x 5] = 16 x 150
		const float* biases,      // [16]
		float* output             // [16 x 8 x 8]
		) {
	int filter_id = blockIdx.z;  // 0–15
	int row = blockIdx.y * blockDim.y + threadIdx.y; // 0–7
	int col = blockIdx.x * blockDim.x + threadIdx.x; // 0–7

	if (row >= 8 || col >= 8) return;

	float sum = 0.0f;

	for (int c = 0; c < CONV2_IN_CHANNELS; c++) {
		for (int i = 0; i < CONV2_FILTER_SIZE; i++) {
			for (int j = 0; j < CONV2_FILTER_SIZE; j++) {
				int in_row = row + i;
				int in_col = col + j;
				int in_idx = c * 12 * 12 + in_row * 12 + in_col;
				int filter_idx = filter_id * (CONV2_IN_CHANNELS * 25) + c * 25 + i * 5 + j;

				sum += input[in_idx] * filters[filter_idx];
			}
		}
	}

	int out_idx = filter_id * CONV2_OUT_SIZE * CONV2_OUT_SIZE + row * CONV2_OUT_SIZE + col;
	output[out_idx] = fmaxf(0.0f, sum + biases[filter_id]); // ReLU
}


__global__ void conv1_kernel(
		const float* input,              // [28 * 28]
		const float* filters,           // [6 * 5 * 5]
		const float* biases,            // [6]
		float* output                   // [6 * 24 * 24]
		) {
	int filter_id = blockIdx.z; // 0 to 5
	int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0 to 23
	int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0 to 23

	if (row >= OUTPUT_SIZE || col >= OUTPUT_SIZE) return;

	float sum = 0.0f;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			int in_row = row + i;
			int in_col = col + j;
			sum += input[in_row * INPUT_SIZE + in_col] *
				filters[filter_id * FILTER_SIZE * FILTER_SIZE + i * FILTER_SIZE + j];
		}
	}

	int output_idx = filter_id * OUTPUT_SIZE * OUTPUT_SIZE + row * OUTPUT_SIZE + col;
	output[output_idx] = fmaxf(0.0f, sum + biases[filter_id]);
}


__global__ void pool1_kernel(
		const float* input,      // [6 * 24 * 24]
		float* output            // [6 * 12 * 12]
		) {
	int fmap = blockIdx.z; // feature map index: 0–5
	int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row: 0–11
	int col = blockIdx.x * blockDim.x + threadIdx.x;  // output col: 0–11

	if (row >= 12 || col >= 12) return;

	int in_row = row * 2;
	int in_col = col * 2;

	float max_val = -1e9;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int idx = fmap * 24 * 24 + (in_row + i) * 24 + (in_col + j);
			float val = input[idx];
			if (val > max_val) max_val = val;
		}
	}

	int out_idx = fmap * 12 * 12 + row * 12 + col;
	output[out_idx] = max_val;
}


// Helper function to read flattened input (28x28 = 784)
void read_input(const char* filename, float* input) {
	FILE* f = fopen(filename, "r");
	for (int i = 0; i < 28 * 28; i++) {
		fscanf(f, "%f", &input[i]);
	}
	fclose(f);
}

// Read filters: 6 filters of size 5x5 (total 6x25)
void read_filters(const char* filename, float* filters) {
	FILE* f = fopen(filename, "r");
	for (int i = 0; i < NUM_FILTERS * FILTER_SIZE * FILTER_SIZE; i++) {
		fscanf(f, "%f", &filters[i]);
	}
	fclose(f);
}

// Read biases: 6 biases
void read_biases(const char* filename, float* biases) {
	FILE* f = fopen(filename, "r");
	for (int i = 0; i < NUM_FILTERS; i++) {
		fscanf(f, "%f", &biases[i]);
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
	for (int iter = 1; iter <= 10; iter++) {
		float *h_input, *h_filters, *h_biases, *h_conv1_output;

		// Allocate host memory
		h_input = (float*)malloc(28 * 28 * sizeof(float));
		h_filters = (float*)malloc(NUM_FILTERS * 25 * sizeof(float));
		h_biases = (float*)malloc(NUM_FILTERS * sizeof(float));
		h_conv1_output = (float*)malloc(NUM_FILTERS * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));


		char input_image_file_path_string[100] = "CUDA_FLATTENED_WEIGHTS/img_";
		char str[20];
		sprintf(str, "%d", iter);
		char txt_extension[10] = ".txt";
		strcat(input_image_file_path_string, str);
		strcat(input_image_file_path_string, txt_extension);
		// Read from files
		read_input(input_image_file_path_string, h_input);
		read_filters("CUDA_FLATTENED_WEIGHTS/conv1_weights.txt", h_filters);
		read_biases("CUDA_FLATTENED_WEIGHTS/conv1_biases.txt", h_biases);




		// Allocate device memory
		float *d_input, *d_filters, *d_biases, *d_output;
		cudaMalloc(&d_input, 28 * 28 * sizeof(float));
		cudaMalloc(&d_filters, NUM_FILTERS * 25 * sizeof(float));
		cudaMalloc(&d_biases, NUM_FILTERS * sizeof(float));
		cudaMalloc(&d_output, NUM_FILTERS * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));

		// Copy data to device
		cudaMemcpy(d_input, h_input, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_filters, h_filters, NUM_FILTERS * 25 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_biases, h_biases, NUM_FILTERS * sizeof(float), cudaMemcpyHostToDevice);

		// Kernel launch config
		dim3 blockDim(16, 16);
		dim3 gridDim((OUTPUT_SIZE + 15) / 16, (OUTPUT_SIZE + 15) / 16, NUM_FILTERS);

		// Launch kernel
		conv1_kernel<<<gridDim, blockDim>>>(d_input, d_filters, d_biases, d_output);
		cudaDeviceSynchronize();


		float* h_pool1_output = (float*)malloc(6 * 12 * 12 * sizeof(float));
		float *d_pool1_output;
		cudaMalloc(&d_pool1_output, 6 * 12 * 12 * sizeof(float));

		dim3 blockDimPool1(16, 16);
		dim3 gridDimPool1((12 + 15) / 16, (12 + 15) / 16, 6);

		pool1_kernel<<<gridDimPool1, blockDimPool1>>>(
				d_output, d_pool1_output  // d_output = Conv1 output
				);
		cudaDeviceSynchronize();



		float* h_conv2_filters = (float*)malloc(16 * 6 * 5 * 5 * sizeof(float));
		float* h_conv2_biases = (float*)malloc(16 * sizeof(float));
		float* h_conv2_output = (float*)malloc(16 * 8 * 8 * sizeof(float));

		read_conv2_filters("CUDA_FLATTENED_WEIGHTS/conv2_weights.txt", h_conv2_filters);
		read_conv2_biases("CUDA_FLATTENED_WEIGHTS/conv2_biases.txt", h_conv2_biases);

		float *d_conv2_filters, *d_conv2_biases, *d_conv2_output;

		cudaMalloc(&d_conv2_filters, 16 * 6 * 5 * 5 * sizeof(float));
		cudaMalloc(&d_conv2_biases, 16 * sizeof(float));
		cudaMalloc(&d_conv2_output, 16 * 8 * 8 * sizeof(float));

		cudaMemcpy(d_conv2_filters, h_conv2_filters, 16 * 6 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_conv2_biases, h_conv2_biases, 16 * sizeof(float), cudaMemcpyHostToDevice);

		dim3 blockDimConv2(8, 8);
		dim3 gridDimConv2((8 + 7) / 8, (8 + 7) / 8, 16);

		conv2_kernel<<<gridDimConv2, blockDimConv2>>>(
				d_pool1_output, d_conv2_filters, d_conv2_biases, d_conv2_output
				);
		cudaDeviceSynchronize();


		float* h_pool2_output = (float*)malloc(16 * 4 * 4 * sizeof(float));
		float* d_pool2_output;
		cudaMalloc(&d_pool2_output, 16 * 4 * 4 * sizeof(float));


		dim3 blockDimPool2(4, 4);  // since output is 4x4
		dim3 gridDimPool2((4 + 3) / 4, (4 + 3) / 4, 16);

		pool2_kernel<<<gridDimPool2, blockDimPool2>>>(
				d_conv2_output, d_pool2_output
				);
		cudaDeviceSynchronize();





		float* h_fc1_weights = (float*)malloc(120 * 256 * sizeof(float));
		float* h_fc1_biases = (float*)malloc(120 * sizeof(float));
		float* h_fc1_output = (float*)malloc(120 * sizeof(float));

		float *d_fc1_weights, *d_fc1_biases, *d_fc1_output, *d_fc1_input;

		read_fc1_weights("CUDA_FLATTENED_WEIGHTS/fc1_weights.txt", h_fc1_weights);
		read_fc1_biases("CUDA_FLATTENED_WEIGHTS/fc1_biases.txt", h_fc1_biases);

		cudaMalloc(&d_fc1_weights, 120 * 256 * sizeof(float));
		cudaMalloc(&d_fc1_biases, 120 * sizeof(float));
		cudaMalloc(&d_fc1_output, 120 * sizeof(float));
		cudaMalloc(&d_fc1_input, 256 * sizeof(float));

		cudaMemcpy(d_fc1_weights, h_fc1_weights, 120 * 256 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_fc1_biases, h_fc1_biases, 120 * sizeof(float), cudaMemcpyHostToDevice);


		dim3 blockDimFlatten(256);
		flatten_pool2<<<1, blockDimFlatten>>>(d_pool2_output, d_fc1_input);



		dim3 blockDimFC1(120);
		dim3 gridDimFC1(1);

		fc1_kernel<<<gridDimFC1, blockDimFC1>>>(
				d_fc1_input, d_fc1_weights, d_fc1_biases, d_fc1_output
				);
		cudaDeviceSynchronize();


		float *h_fc2_weights = (float*)malloc(84 * 120 * sizeof(float));
		float *h_fc2_biases  = (float*)malloc(84 * sizeof(float));
		float *h_fc2_output  = (float*)malloc(84 * sizeof(float));

		float *d_fc2_weights, *d_fc2_biases, *d_fc2_output;

		read_fc2_weights("CUDA_FLATTENED_WEIGHTS/fc2_weights.txt", h_fc2_weights);
		read_fc2_biases("CUDA_FLATTENED_WEIGHTS/fc2_biases.txt", h_fc2_biases);

		cudaMalloc(&d_fc2_weights, 84 * 120 * sizeof(float));
		cudaMalloc(&d_fc2_biases,  84 * sizeof(float));
		cudaMalloc(&d_fc2_output,  84 * sizeof(float));

		cudaMemcpy(d_fc2_weights, h_fc2_weights, 84 * 120 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_fc2_biases,  h_fc2_biases,  84 * sizeof(float), cudaMemcpyHostToDevice);

		dim3 blockDimFC2(84);
		dim3 gridDimFC2(1);

		fc2_kernel<<<gridDimFC2, blockDimFC2>>>(d_fc1_output, d_fc2_weights, d_fc2_biases, d_fc2_output);
		cudaDeviceSynchronize();




		float *h_fc3_weights = (float*)malloc(10 * 84 * sizeof(float));
		float *h_fc3_biases  = (float*)malloc(10 * sizeof(float));
		float *h_fc3_output  = (float*)malloc(10 * sizeof(float));

		float *d_fc3_weights, *d_fc3_biases, *d_fc3_output;

		read_fc3_weights("CUDA_FLATTENED_WEIGHTS/fc3_weights.txt", h_fc3_weights);
		read_fc3_biases("CUDA_FLATTENED_WEIGHTS/fc3_biases.txt", h_fc3_biases);

		cudaMalloc(&d_fc3_weights, 10 * 84 * sizeof(float));
		cudaMalloc(&d_fc3_biases,  10 * sizeof(float));
		cudaMalloc(&d_fc3_output,  10 * sizeof(float));

		cudaMemcpy(d_fc3_weights, h_fc3_weights, 10 * 84 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_fc3_biases,  h_fc3_biases,  10 * sizeof(float), cudaMemcpyHostToDevice);

		dim3 blockDimFC3(10);
		dim3 gridDimFC3(1);

		fc3_kernel<<<gridDimFC3, blockDimFC3>>>(d_fc2_output, d_fc3_weights, d_fc3_biases, d_fc3_output);
		cudaDeviceSynchronize();


		float* d_probabilities;
		float* h_probabilities = (float*)malloc(10 * sizeof(float));
		cudaMalloc(&d_probabilities, 10 * sizeof(float));

		dim3 blockDimSoftmax(10);
		dim3 gridDimSoftmax(1);
		softmax_kernel<<<gridDimSoftmax, blockDimSoftmax>>>(d_fc3_output, d_probabilities, 10);
		cudaDeviceSynchronize();

		//Uncomment the below lines of code to extract the output of each kernel into text files, to use it in
		//Python code to verify the code implementation correctness
		/*
		// Copy result back
		cudaMemcpy(h_conv1_output, d_output, NUM_FILTERS * OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

		// Save output
		save_output("conv1_cuda_output.txt", h_conv1_output);


		cudaMemcpy(h_pool1_output, d_pool1_output, 6 * 12 * 12 * sizeof(float), cudaMemcpyDeviceToHost);
		save_pool1_output("pool1_cuda_output.txt", h_pool1_output);

		cudaMemcpy(h_conv2_output, d_conv2_output, 16 * 8 * 8 * sizeof(float), cudaMemcpyDeviceToHost);
		save_conv2_output("conv2_cuda_output.txt", h_conv2_output);


		cudaMemcpy(h_pool2_output, d_pool2_output, 16 * 4 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
		save_pool2_output("pool2_cuda_output.txt", h_pool2_output);



		cudaMemcpy(h_fc1_output, d_fc1_output, 120 * sizeof(float), cudaMemcpyDeviceToHost);
		save_fc1_output("fc1_cuda_output.txt", h_fc1_output);

		cudaMemcpy(h_fc2_output, d_fc2_output, 84 * sizeof(float), cudaMemcpyDeviceToHost);
		save_fc2_output("fc2_cuda_output.txt", h_fc2_output);


		cudaMemcpy(h_fc3_output, d_fc3_output, 10 * sizeof(float), cudaMemcpyDeviceToHost);
		save_fc3_output("fc3_cuda_output.txt", h_fc3_output);
		 */

		cudaMemcpy(h_probabilities, d_probabilities, 10 * sizeof(float), cudaMemcpyDeviceToHost);

		// Print predicted class
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


		// Free memory
		//    free(h_input); free(h_filters); free(h_biases); free(h_conv1_output);
		//  cudaFree(d_input); cudaFree(d_filters); cudaFree(d_biases); cudaFree(d_output);

		free(h_input);
		free(h_filters);
		free(h_biases);
		free(h_conv1_output);
		free(h_pool1_output);
		free(h_conv2_filters);
		free(h_conv2_biases);
		free(h_conv2_output);
		free(h_pool2_output);
		free(h_fc1_weights);
		free(h_fc1_biases);
		free(h_fc1_output);
		free(h_fc2_weights);
		free(h_fc2_biases);
		free(h_fc2_output);
		free(h_fc3_weights);
		free(h_fc3_biases);
		free(h_fc3_output);
		free(h_probabilities);

		// Free device memory
		cudaFree(d_input);
		cudaFree(d_filters);
		cudaFree(d_biases);
		cudaFree(d_output);
		cudaFree(d_pool1_output);
		cudaFree(d_conv2_filters);
		cudaFree(d_conv2_biases);
		cudaFree(d_conv2_output);
		cudaFree(d_pool2_output);
		cudaFree(d_fc1_weights);
		cudaFree(d_fc1_biases);
		cudaFree(d_fc1_output);
		cudaFree(d_fc1_input);
		cudaFree(d_fc2_weights);
		cudaFree(d_fc2_biases);
		cudaFree(d_fc2_output);
		cudaFree(d_fc3_weights);
		cudaFree(d_fc3_biases);
		cudaFree(d_fc3_output);
		cudaFree(d_probabilities);

		cudaCheckError();

	}
	return 0;
}
