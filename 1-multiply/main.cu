#include "consts.hpp"
#include <iostream>

__global__ auto kernel_multiply(float* arr, float multiplier) -> void {
    size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (idx < ARRAY_LEN)
        arr[idx] *= multiplier;
}

auto cuda_print_err(const cudaError err) -> void {
    if (err != cudaSuccess)
        std::cout << "CUDA Error [" << __LINE__ << "] - '"
                  << cudaGetErrorString(err) << "'\n";
}

auto cuda_run_multiply(float* arr, const float multiplier) -> void {
    constexpr size_t THREADS = 128;
    constexpr size_t BLOCKS = (ARRAY_LEN + THREADS - 1) / THREADS;

    float* cuda_arr = nullptr;
    cudaError err = cudaMalloc(&cuda_arr, ARRAY_SIZE);
    cuda_print_err(err);

    err = cudaMemcpy(cuda_arr, arr, ARRAY_SIZE, cudaMemcpyHostToDevice);
    cuda_print_err(err);

    kernel_multiply<<<BLOCKS, THREADS>>>(cuda_arr, multiplier);
    cuda_print_err(cudaGetLastError());

    err = cudaMemcpy(arr, cuda_arr, ARRAY_SIZE, cudaMemcpyDeviceToHost);
    cuda_print_err(err);

    cudaFree(cuda_arr);
}
