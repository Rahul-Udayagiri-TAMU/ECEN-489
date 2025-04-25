#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

void save_conv1_output(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int f_idx = 0; f_idx < CONV1_FILTERS; f_idx++) {
		for (int i = 0; i < 24; i++) {
			for (int j = 0; j < 24; j++) {
				fprintf(f, "%.6f ", conv1_output[f_idx][i][j]);
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void save_pool1_output(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int f_idx = 0; f_idx < CONV1_FILTERS; f_idx++) {
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				fprintf(f, "%.6f ", pool1_output[f_idx][i][j]);
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void save_conv2_output(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int f_idx = 0; f_idx < CONV2_FILTERS; f_idx++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				fprintf(f, "%.6f ", conv2_output[f_idx][i][j]);
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void save_pool2_output(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int f_idx = 0; f_idx < CONV2_FILTERS; f_idx++) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				fprintf(f, "%.6f ", pool2_output[f_idx][i][j]);
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void save_flattened_input_into_output_file(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < 256; i++) {
		fprintf(f, "%.6f\n", fc_input[i]);
	}
	fclose(f);
}

void save_fc1_output(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < FC1_SIZE; i++) {
		fprintf(f, "%.6f\n", fc1_output[i]);
	}
	fclose(f);
}

void save_fc2_output(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < FC2_SIZE; i++) {
		fprintf(f, "%.6f\n", fc2_output[i]);
	}
	fclose(f);
}

void save_fc3_output(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < FC3_SIZE; i++) {
		fprintf(f, "%.6f\n", fc3_output[i]);
	}
	fclose(f);
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



void save_input_image_flat(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < IMG_SIZE; i++) {
		for (int j = 0; j < IMG_SIZE; j++) {
			fprintf(f, "%.6f\n", input_image[i][j]);
		}
	}
	fclose(f);
}

void save_conv1_weights_flat(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int f_idx = 0; f_idx < CONV1_FILTERS; f_idx++) {
		for (int i = 0; i < CONV1_KERNEL_SIZE; i++) {
			for (int j = 0; j < CONV1_KERNEL_SIZE; j++) {
				fprintf(f, "%.6f\n", conv1_weights[f_idx][i][j]);
			}
		}
	}
	fclose(f);
}

void save_conv1_biases(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < CONV1_FILTERS; i++) {
		fprintf(f, "%.6f\n", conv1_biases[i]);
	}
	fclose(f);
}

void save_conv2_weights_flat(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int f_idx = 0; f_idx < CONV2_FILTERS; f_idx++) {
		for (int c = 0; c < CONV1_FILTERS; c++) {  // 6 input channels
			for (int i = 0; i < CONV2_KERNEL_SIZE; i++) {
				for (int j = 0; j < CONV2_KERNEL_SIZE; j++) {
					fprintf(f, "%.6f\n", conv2_weights[f_idx][c][i][j]);
				}
			}
		}
	}
	fclose(f);
}

void save_conv2_biases(const char* filename) {
	FILE* f = fopen(filename, "w");
	for (int i = 0; i < CONV2_FILTERS; i++) {
		fprintf(f, "%.6f\n", conv2_biases[i]);
	}
	fclose(f);
}

void save_fc1_weights_flat(const char* filename) {
	FILE* f = fopen(filename, "w");
	if (!f) {
		printf("Error: Could not open %s for writing\n", filename);
		return;
	}

	for (int i = 0; i < 120; ++i) {
		for (int j = 0; j < 256; ++j) {
			fprintf(f, "%.6f\n", fc1_weights[i][j]);
		}
	}

	fclose(f);
}

void save_fc1_biases(const char* filename) {
	FILE* f = fopen(filename, "w");
	if (!f) {
		printf("Error: Could not open %s for writing\n", filename);
		return;
	}

	for (int i = 0; i < 120; ++i) {
		fprintf(f, "%.6f\n", fc1_biases[i]);
	}

	fclose(f);
}

void save_fc2_weights_flat(const char* filename) {
	FILE* f = fopen(filename, "w");
	if(!f) {
		printf("Error: Could not open %s for writing\n", filename);
		return;
	}
	for (int i = 0; i < 84; i++) {
		for (int j = 0; j < 120; j++) {
			fprintf(f, "%.6f\n", fc2_weights[i][j]);
		}
	}

	fclose(f);
}


void save_fc2_biases(const char* filename) {
	FILE* f = fopen(filename, "w");
	if(!f) {
		printf("Error: Could not open %s for writing\n", filename);
		return;
	}
	for (int i = 0; i < 84; i++) {
		fprintf(f, "%.6f\n", fc2_biases[i]);
	}
	fclose(f);
}

void save_fc3_weights_flat(const char* filename) {
	FILE* f = fopen(filename, "w");
	if(!f) {
		printf("Error: Could not open %s for writing\n", filename);
		return;
	}
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 84; j++) {
			fprintf(f, "%.6f\n", fc3_weights[i][j]);
		}
	}

	fclose(f);
}


void save_fc3_biases(const char* filename) {
	FILE* f = fopen(filename, "w");
	if(!f) {
		printf("Error: Could not open %s for writing\n", filename);
		return;
	}
	for (int i = 0; i < 10; i++) {
		fprintf(f, "%.6f\n", fc3_biases[i]);
	}
	fclose(f);
}


int main(int argc, char** argv) {
	for (int iter = 1; iter <= 10; iter++) {
		/*	    if (argc != 2) {
			    printf("Usage: %s <28x28_grayscale_image.png>\n", argv[0]);
			    return 1;
			    }
			    */

		char str[20];
		sprintf(str, "%d", iter);

		char input_image_string[100] = "TestDataSet/img_";
		char jpg_extension[10] = ".jpg";
		strcat(input_image_string, str);
		strcat(input_image_string, jpg_extension);

		// To load the weights extracted from Python code
		load_weights("weights.txt");
		load_input_image(input_image_string);

		// If need the data for flattened images to be utilized in CUDA program, run the code by uncommenting the below block	    
		/*

		//Flatten the read data for CUDA programs to utilize
		char export_image_string[100] = "CUDA_FLATTENED_WEIGHTS/img_";
		char txt_extension[10] = ".txt";	    
		strcat(export_image_string, str);
		strcat(export_image_string, txt_extension);
		save_input_image_flat(export_image_string);
		if(iter == 1) {
		save_conv1_weights_flat("CUDA_FLATTENED_WEIGHTS/conv1_weights.txt");
		save_conv1_biases("CUDA_FLATTENED_WEIGHTS/conv1_biases.txt");

		save_conv2_weights_flat("CUDA_FLATTENED_WEIGHTS/conv2_weights.txt");
		save_conv2_biases("CUDA_FLATTENED_WEIGHTS/conv2_biases.txt");

		save_fc1_weights_flat("CUDA_FLATTENED_WEIGHTS/fc1_weights.txt");
		save_fc1_biases("CUDA_FLATTENED_WEIGHTS/fc1_biases.txt");

		save_fc2_weights_flat("CUDA_FLATTENED_WEIGHTS/fc2_weights.txt");
		save_fc2_biases("CUDA_FLATTENED_WEIGHTS/fc2_biases.txt");

		save_fc3_weights_flat("CUDA_FLATTENED_WEIGHTS/fc3_weights.txt");
		save_fc3_biases("CUDA_FLATTENED_WEIGHTS/fc3_biases.txt");
		}
		*/


		float probabilities[10];
		
		time_t start1 = clock();
		// Forward Pass Implementation
		conv1_forward();
		time_t start2 = clock();
		pool1_forward();
		clock_t start3 = clock();
		conv2_forward();
		clock_t start4 = clock();
		pool2_forward();
		clock_t start5 = clock();
		flatten_pool2_output();
		clock_t start6 = clock();
		fc1_forward();
		clock_t start7 = clock();
		fc2_forward();
		clock_t start8 = clock();
		fc3_forward();
		clock_t start9 = clock();
		softmax(fc3_output, probabilities, 10);
		clock_t end = clock();

		double cpu_time_ms;

		printf("For Image - %s\n", input_image_string); 
	
		cpu_time_ms = ((double)(start2 - start1) / CLOCKS_PER_SEC) * 1000.0;
		printf("CONV1 Execution Time (CPU): %.15f ms\n", cpu_time_ms);
		
		cpu_time_ms = ((double)(start3 - start2) / CLOCKS_PER_SEC) * 1000.0;
		printf("Pool1 Execution Time (CPU): %.9f ms\n", cpu_time_ms);

		cpu_time_ms = ((double)(start4 - start3) / CLOCKS_PER_SEC) * 1000.0;
		printf("CONV2 Execution Time (CPU): %.9f ms\n", cpu_time_ms);
		
		cpu_time_ms = ((double)(start5 - start4) / CLOCKS_PER_SEC) * 1000.0;
		printf("Pool2 Execution Time (CPU): %.9f ms\n", cpu_time_ms);
		
		cpu_time_ms = ((double)(start6 - start5) / CLOCKS_PER_SEC) * 1000.0;
		printf("Flatten Execution Time (CPU): %.9f ms\n", cpu_time_ms);
	       	
		cpu_time_ms = ((double)(start7 - start6) / CLOCKS_PER_SEC) * 1000.0;
		printf("FC1 Execution Time (CPU): %.9f ms\n", cpu_time_ms);

	       	cpu_time_ms = ((double)(start8 - start7) / CLOCKS_PER_SEC) * 1000.0;
		printf("FC2 Execution Time (CPU): %.9f ms\n", cpu_time_ms);
	       		
		cpu_time_ms = ((double)(start9 - start8) / CLOCKS_PER_SEC) * 1000.0;
		printf("FC3 Execution Time (CPU): %.9f ms\n", cpu_time_ms);

	       	cpu_time_ms = ((double)(end - start9) / CLOCKS_PER_SEC) * 1000.0;
		printf("Softmax Execution Time (CPU): %.9f ms\n", cpu_time_ms);

		//If need to verify the implementation correctness of this code using Python, uncomment the below block, extract the outputs
		//and pass these as inputs to the Python scripts
		/*
		   save_conv1_output("conv1_output.txt");
		   save_pool1_output("pool1_output.txt");
		   save_conv2_output("conv2_output.txt");
		   save_pool2_output("pool2_output.txt");
		   save_flattened_input_into_output_file("flattened_input_output.txt");
		   save_fc1_output("fc1_output.txt");
		   save_fc2_output("fc2_output.txt");
		   save_fc3_output("fc3_output.txt");
		   */




		int prediction = 0;
		float max_probability = probabilities[0];
		for (int i = 0; i < 10; i++) {
			if(probabilities[i] > max_probability)
			{
				prediction = i;
				max_probability = probabilities[i];
			}
		}
		printf("Predicted Digit: %d\n", prediction);


		for (int i = 0; i < 10; ++i) {
			printf("Class %d: %.4f\n", i, probabilities[i]);
		} 
	}
	return 0;
}
