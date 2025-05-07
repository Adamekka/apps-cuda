#include "cuda_image.hpp"
#include "uni_mem_allocator.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

auto cuda_run_insert_image(
    const CudaImage& big_image,
    const CudaImage& small_image,
    const CudaImage& result_image
) -> void;

auto cuda_run_rotate_image(
    const CudaImage& input_image,
    const CudaImage& result_image,
    float angle_radians
) -> void;

auto cuda_run_bilinear_scale(
    const CudaImage& input_image, const CudaImage& result_image
) -> void;

auto main(int32_t argc, char** argv) -> int32_t {
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (argc < 3) {
        std::cout << "Usage: <big-image-path> <small-image-path>\n";
        return 1;
    }

    cv::Mat big_image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    assert(big_image.channels() == 4);

    if (!big_image.data) {
        std::cout << "Unable to read file '" << argv[1] << "'\n";
        return 1;
    }

    cv::Mat small_image = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    assert(big_image.channels() == 4);

    if (!small_image.data) {
        std::cout << "Unable to read file '" << argv[2] << "'\n";
        return 1;
    }

    assert(big_image.size().width >= small_image.size().width);
    assert(big_image.size().height >= small_image.size().height);

    cv::Mat insert_image = cv::Mat(big_image.size(), CV_8UC4);

    CudaImage cuda_big_image{
        .size = {static_cast<uint32_t>(big_image.size().width), static_cast<uint32_t>(big_image.size().height)},
        .data_uchar4 = reinterpret_cast<uchar4*>(big_image.data), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    CudaImage cuda_small_image{
        .size = {static_cast<uint32_t>(small_image.size().width), static_cast<uint32_t>(small_image.size().height)},
        .data_uchar4 = reinterpret_cast<uchar4*>(small_image.data), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    CudaImage cuda_insert_image{
        .size = {static_cast<uint32_t>(insert_image.size().width), static_cast<uint32_t>(insert_image.size().height)},
        .data_uchar4 = reinterpret_cast<uchar4*>(insert_image.data), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    cuda_run_insert_image(cuda_big_image, cuda_small_image, cuda_insert_image);

    cv::Mat rotated_image = cv::Mat(insert_image.size(), CV_8UC4);

    CudaImage cuda_rotated_image{
        .size = {static_cast<uint32_t>(rotated_image.size().width), static_cast<uint32_t>(rotated_image.size().height)},
        .data_uchar4 = reinterpret_cast<uchar4*>(rotated_image.data), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    cuda_run_rotate_image(cuda_insert_image, cuda_rotated_image, 3);

    cv::Mat bilin_scale_image
        = cv::Mat(insert_image.rows / 2, insert_image.cols * 2, CV_8UC4);

    CudaImage cuda_bilin_scale_image{
        .size = {static_cast<uint32_t>(bilin_scale_image.size().width), static_cast<uint32_t>(bilin_scale_image.size().height)},
        .data_uchar4 = reinterpret_cast<uchar4*>(bilin_scale_image.data), // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    cuda_run_bilinear_scale(cuda_rotated_image, cuda_bilin_scale_image);

    cv::imshow("Big", big_image);
    cv::imshow("Small", small_image);
    cv::imshow("Insert", insert_image);
    cv::imshow("Rotated", rotated_image);
    cv::imshow("Bilinear scale", bilin_scale_image);
    cv::waitKey();
}
