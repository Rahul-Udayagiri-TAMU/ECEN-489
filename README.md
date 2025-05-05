# CUDA-Based LeNet-5 Inference Implementation

**Course:** ECEN-489 – GPU Programming Visualization
**Project:** Accelerating CNN Inference with CUDA

This project implements a forward-pass (inference) pipeline for a LeNet-5 Convolutional Neural Network using CUDA. It compares CPU (C) and GPU (CUDA) implementations, including versions with global and shared memory optimizations.

## Features
- Full model: Conv1 → Pool1 → Conv2 → Pool2 → FC1 → FC2 → FC3 → Softmax
- Global and shared memory variants for key layers
- Output verification and performance benchmarking
- Speedup analysis compared to CPU implementation
