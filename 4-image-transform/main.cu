#include "cuda_image.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>

// MARK: Insert image

__global__ auto kernel_insert_image(
    CudaImage big_image, CudaImage small_image, CudaImage result_image
) -> void {
    uint32_t x = (blockDim.x * blockIdx.x) + threadIdx.x;
    uint32_t y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (x >= big_image.size.x || y >= big_image.size.y)
        return;

    uint32_t inset_x = (big_image.size.x - small_image.size.x) / 2;
    uint32_t inset_y = (big_image.size.y - small_image.size.y) / 2;

    // Check if kernel index is in range of small image
    bool inserting_small_image
        = inset_x <= x && x < small_image.size.x + inset_x && inset_y <= y
       && y < small_image.size.y + inset_y;

    uchar3 image_pixel
        = inserting_small_image
            ? small_image.data_uchar3
                  [((y - inset_y) * small_image.size.x) + x - inset_x]
            : big_image.data_uchar3[(y * big_image.size.x) + x];

    result_image.data_uchar3[(y * result_image.size.x) + x] = image_pixel;
}

auto cuda_run_insert_image(
    const CudaImage& big_image,
    const CudaImage& small_image,
    const CudaImage& result_image
) -> void {
    constexpr size_t BLOCK_SIZE = 16;

    dim3 blocks(
        (big_image.size.x + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (big_image.size.y + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    kernel_insert_image<<<blocks, threads>>>(
        big_image, small_image, result_image
    );

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA Error [" << __LINE__ << "] - '"
                  << cudaGetErrorString(err) << "'\n";

    cudaDeviceSynchronize();
}

// MARK: Rotate image

__device__ auto sample_or_black(CudaImage img, uint32_t x, uint32_t y)
    -> uchar3 {
    if (x >= img.size.x || y >= img.size.y)
        return make_uchar3(0, 0, 0);

    return img.data_uchar3[(y * img.size.x) + x];
}

__global__ void kernel_rotate_image(
    CudaImage input_image, CudaImage result_image, float angle_radians
) {
    auto x_rotate
        = static_cast<int32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
    auto y_rotate
        = static_cast<int32_t>((blockIdx.y * blockDim.y) + threadIdx.y);

    if (x_rotate >= input_image.size.x || y_rotate >= input_image.size.y)
        return;

    float sin = sinf(angle_radians);
    float cos = cosf(angle_radians);

    int32_t rotate_cx = static_cast<int32_t>(result_image.size.x) / 2;
    int32_t rotate_cy = static_cast<int32_t>(result_image.size.y) / 2;

    int32_t x_centered = x_rotate - rotate_cx;
    int32_t y_centered = y_rotate - rotate_cy;

    float x_orig_f = (cos * static_cast<float>(x_centered))
                   + (sin * static_cast<float>(y_centered));
    float y_orig_f = (-sin * static_cast<float>(x_centered))
                   + (cos * static_cast<float>(y_centered));

    int32_t orig_cx = static_cast<int32_t>(input_image.size.x) / 2;
    int32_t orig_cy = static_cast<int32_t>(input_image.size.y) / 2;

    auto x_orig
        = static_cast<int32_t>(roundf(x_orig_f + static_cast<float>(orig_cx)));
    auto y_orig
        = static_cast<int32_t>(roundf(y_orig_f + static_cast<float>(orig_cy)));

    if (x_orig < 0 || x_orig >= input_image.size.x || y_orig < 0
        || y_orig >= input_image.size.y)
        return;

    uchar3 pixel
        = input_image.data_uchar3[(y_orig * input_image.size.x) + x_orig];
    result_image.data_uchar3[(y_rotate * result_image.size.x) + x_rotate]
        = pixel;
}

auto cuda_run_rotate_image(
    const CudaImage& input_image,
    const CudaImage& result_image,
    float angle_radians
) -> void {
    constexpr size_t BLOCK_SIZE = 16;

    dim3 blocks(
        (input_image.size.x + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (input_image.size.y + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    kernel_rotate_image<<<blocks, threads>>>(
        input_image, result_image, angle_radians
    );

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA Error [" << __LINE__ << "] - '"
                  << cudaGetErrorString(err) << "'\n";

    cudaDeviceSynchronize();
}

// MARK: Bilinear interpolate

__device__ auto bilinear_interpolate(const CudaImage img, float x, float y)
    -> uchar3 {
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = x - static_cast<float>(x0);
    float dy = y - static_cast<float>(y0);

    auto get = [&](int xi, int yi) -> uchar3 {
        if (xi < 0 || xi >= img.size.x || yi < 0 || yi >= img.size.y)
            return make_uchar3(0, 0, 0);

        return img.data_uchar3[(yi * img.size.x) + xi];
    };

    uchar3 p00 = get(x0, y0);
    uchar3 p01 = get(x1, y0);
    uchar3 p10 = get(x0, y1);
    uchar3 p11 = get(x1, y1);

    uchar3 result;
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    for (int i = 0; i < 3; ++i) {
        float v00 = (reinterpret_cast<uint8_t*>(&p00))[i];
        float v01 = (reinterpret_cast<uint8_t*>(&p01))[i];
        float v10 = (reinterpret_cast<uint8_t*>(&p10))[i];
        float v11 = (reinterpret_cast<uint8_t*>(&p11))[i];

        float value = (v00 * (1 - dx) * (1 - dy)) + (v01 * dx * (1 - dy))
                    + (v10 * (1 - dx) * dy) + (v11 * dx * dy);

        (reinterpret_cast<uint8_t*>(&result))[i] = static_cast<uint8_t>(value);
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

    return result;
}

__global__ auto
kernel_bilinear_scale(CudaImage input_image, CudaImage result_image) -> void {
    uint32_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= result_image.size.x || y >= result_image.size.y)
        return;

    float scale_x = static_cast<float>(input_image.size.x - 1)
                  / static_cast<float>(result_image.size.x);
    float scale_y = static_cast<float>(input_image.size.y - 1)
                  / static_cast<float>(result_image.size.y);

    float src_x = static_cast<float>(x) * scale_x;
    float src_y = static_cast<float>(y) * scale_y;

    uchar3 interpolated = bilinear_interpolate(input_image, src_x, src_y);
    result_image.data_uchar3[(y * result_image.size.x) + x] = interpolated;
}

auto cuda_run_bilinear_scale(
    const CudaImage& input_image, const CudaImage& result_image
) -> void {
    constexpr size_t BLOCK_SIZE = 16;

    dim3 blocks(
        (result_image.size.x + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (result_image.size.y + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    kernel_bilinear_scale<<<blocks, threads>>>(input_image, result_image);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "CUDA Error [" << __LINE__ << "] - '"
                  << cudaGetErrorString(err) << "'\n";

    cudaDeviceSynchronize();
}
