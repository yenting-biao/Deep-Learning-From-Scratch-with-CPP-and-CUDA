#ifndef CUDA_ERROR_HANDLER_HPP
#define CUDA_ERROR_HANDLER_HPP

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK_ERROR(cudaFunctionCall)                 \
    do {                                                   \
        cudaError_t result = cudaFunctionCall;             \
        if (result != cudaSuccess) {                       \
            fprintf(stderr, "A CUDA error occurred: %s\n", \
                    cudaGetErrorString(result));           \
            exit(1);                                       \
        }                                                  \
    } while (0)

#endif  // CUDA_ERROR_HANDLER_HPP