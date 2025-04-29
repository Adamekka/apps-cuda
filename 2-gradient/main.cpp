#include "consts.hpp"
#include "cuda_image.hpp"
#include "uni_mem_allocator.h"
#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

auto cuda_run_gradient(const CudaImage& image) -> void;

auto main() -> int32_t {
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    cv::Mat image(SIZE_Y, SIZE_X, CV_8UC3);

    for (int32_t y = 0; y < image.rows; y++) {
        for (int32_t x = 0; x < image.cols; x++) {
            int32_t dx = x - (image.cols / 2);

            uint8_t grad = 255 * abs(dx) / (image.cols / 2);
            uint8_t inv_grad = 255 - grad;

            uchar3 background = (dx < 0) ? (uchar3){grad, inv_grad, 0}
                                         : (uchar3){0, inv_grad, grad};

            cv::Vec3b background_vec(background.x, background.y, background.z);
            image.at<cv::Vec3b>(y, x) = background_vec;
        }
    }

    cv::imshow("B-G-R Gradient", image);

    CudaImage cuda_image{
      .size = {static_cast<uint32_t>(image.size().width), static_cast<uint32_t>(image.size().height)},
      .data_uchar3 = reinterpret_cast<uchar3*>(image.data), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    cuda_run_gradient(cuda_image);

    cv::imshow("B-G-R Gradient & Color Rotation", image);
    cv::waitKey();
}
