#include "consts.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

auto cuda_run_multiply(float* arr, float multiplier) -> void;

auto main() -> int32_t {
    float arr[ARRAY_LEN]; // NOLINT(cppcoreguidelines-avoid-c-arrays)

    for (size_t i = 0; i < ARRAY_LEN; ++i)
        arr[i] = static_cast<float>(i);

    cuda_run_multiply(static_cast<float*>(arr), M_PI);

    for (const auto& val : arr)
        std::cout << std::fixed << std::setprecision(2) << std::setw(8) << val;

    std::cout << '\n';
}
