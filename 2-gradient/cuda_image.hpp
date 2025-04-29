#ifndef CUDA_IMAGE_HPP
#define CUDA_IMAGE_HPP

#include <vector_types.h>

struct CudaImage {
    uint3 size;

    union {
        void* data_void;
        uchar1* data_uchar1;
        uchar3* data_uchar3;
        uchar4* data_uchar4;
    };
};

#endif
