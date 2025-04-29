#include "consts.hpp"
#include "cuda_image.hpp"
#include <cstdint>
#include <iostream>

__global__ auto kernel_gradient(CudaImage image) -> void {
    uint32_t x = (blockDim.x * blockIdx.x) + threadIdx.x;
    uint32_t y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (x >= image.size.x || y >= image.size.y)
        return;

    uchar3 background;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    uchar3 tmp = image.data_uchar3[(y * image.size.x) + x];

    background.x = tmp.y;
    background.y = tmp.z;
    background.z = tmp.x;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    image.data_uchar3[(y * image.size.x) + x] = background;
}

auto cuda_run_gradient(const CudaImage& image) -> void {
    dim3 grid(
        (image.size.x + BLOCK_X - 1) / BLOCK_X,
        (image.size.y + BLOCK_Y - 1) / BLOCK_Y
    );

    kernel_gradient<<<grid, dim3(BLOCK_X, BLOCK_Y)>>>(image);
    cudaError err = cudaGetLastError();

    if (err != cudaSuccess)
        std::cout << "CUDA Error [" << __LINE__ << "] - '"
                  << cudaGetErrorString(err) << "'\n";

    cudaDeviceSynchronize();
}
