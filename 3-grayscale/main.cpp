#include "cuda_image.hpp"
#include "uni_mem_allocator.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

auto cuda_run_grayscale(const CudaImage& image, const CudaImage& bw_image)
    -> void;

auto main(int32_t argc, char** argv) -> int32_t {
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (argc < 2) {
        std::cout << "Enter picture filename!\n";
        return 1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

    if (!image.data) {
        std::cout << "Unable to read file '" << argv[1] << "'\n";
        return 1;
    }

    cv::Mat bw_image(image.size(), CV_8UC3);

    CudaImage cuda_image{
        .size = {static_cast<uint32_t>(image.size().width), static_cast<uint32_t>(image.size().height)},
        .data_uchar3 = reinterpret_cast<uchar3*>(image.data), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    CudaImage cuda_bw_image{
        .size = {static_cast<uint32_t>(bw_image.size().width), static_cast<uint32_t>(bw_image.size().height)},
        .data_uchar3 = reinterpret_cast<uchar3*>(bw_image.data), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    cuda_run_grayscale(cuda_image, cuda_bw_image);

    cv::imshow("Color", image);
    cv::imshow("GrayScale", bw_image);
    cv::waitKey();
}
