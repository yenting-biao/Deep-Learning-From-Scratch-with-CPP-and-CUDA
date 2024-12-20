#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

void softmax(double* out, const double* const in, int size);
void transposeMatrix(double** out, double** in, int row, int col);
void multiplyMatrix(double** out, double** a, double** b, int row_a, int col_a,
                    int row_b, int col_b);
double** create2DArray(int row, int col);
void free2DArray(double** arr, int row);
void print_progress_bar(
    int current_epoch, int num_epochs, int current_batch, int num_batches,
    double current_loss,
    std::chrono::_V2::system_clock::time_point epoch_start_time,
    double mean_loss = 0);

void copy2DArrayToDevice(double** dest_d, double** src_h, const int row,
                         const int col);
void copy2DArrayFromDevice(double** dest_h, double** src_d, const int row,
                           const int col);
double** create2DArrayCuda(int row, int col, double** copy_src = nullptr);
double*** create3DArrayCuda(const int row, const int col, const int depth);
void free2DArrayCuda(double** arr, int row, double** copy_image = nullptr,
                     int col = 0);
void free3DArrayCuda(double*** arr, const int row, const int col);

void softmaxCuda(double** out, double** in, int len, int batch_size);

__device__ double atomicAddDouble(double* address, double val);

#endif  // UTILS_HPP
