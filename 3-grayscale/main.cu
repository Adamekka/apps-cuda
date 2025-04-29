#include "cuda_image.hpp"
#include <cstdint>
#include <iostream>

__global__ auto kernel_grayscale(CudaImage color_image, CudaImage bw_image)
    -> void {
    uint32_t x = (blockDim.x * blockIdx.x) + threadIdx.x;
    uint32_t y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (x >= color_image.size.x || y >= color_image.size.y)
        return;

    uchar3 background = color_image.data_uchar3[(y * color_image.size.x) + x];

    background.x = static_cast<uint8_t>(
        (background.x * 0.11) + (background.y * 0.59) + (background.z * 0.30)
    );

    // if (background.x < 100)
    //     background.x = 0;
    // else
    //     background.x -= 100;

    // if (background.y < 100)
    //     background.y = 0;
    // else
    //     background.y -= 100;

    // if (background.z < 100)
    //     background.z = 0;
    // else
    //     background.z -= 100;

    bw_image.data_uchar3[(y * bw_image.size.x) + x].x = background.x;
    // bw_image.data_uchar3[(y * bw_image.size.x) + x].y = background.y;
    // bw_image.data_uchar3[(y * bw_image.size.x) + x].z = background.z;
}

auto cuda_run_grayscale(const CudaImage& image, const CudaImage& bw_image)
    -> void {
    constexpr size_t BLOCK_SIZE = 16;

    dim3 blocks(
        (image.size.x + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (image.size.y + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    kernel_grayscale<<<blocks, threads>>>(image, bw_image);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA Error [" << __LINE__ << "] - '"
                  << cudaGetErrorString(err) << "'\n";

    cudaDeviceSynchronize();
}
