#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include "utils.hpp"

void softmax(double* const softmax_output, const double* const logits,
             const int n) {
    const double epsilon = 1e-12;

    double max_logit = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n; i++) {
        softmax_output[i] = exp(logits[i] - max_logit) + epsilon;
        sum_exp += softmax_output[i];
    }
    // Normalize the softmax output
    for (int i = 0; i < n; i++) {
        softmax_output[i] /= sum_exp;
    }
}

void transposeMatrix(double** const out, double** const in, const int row,
                     const int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            out[j][i] = in[i][j];
        }
    }
}

void multiplyMatrix(double** const out, double** const a, double** const b,
                    const int row_a, const int col_a, const int row_b,
                    const int col_b) {
    if (col_a != row_b) {
        std::cerr << "<ERROR> multiplyMatrix(): Invalid matrix dimensions for "
                     "multiplication: "
                  << row_a << " x " << col_a << " cannot multiply with "
                  << row_b << " x " << col_b << std::endl;
        exit(1);
    }

    for (int i = 0; i < row_a; i++) {
        for (int j = 0; j < col_b; j++) {
            out[i][j] = 0.0;
            for (int k = 0; k < col_a; k++) {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

double** create2DArray(const int row, const int col) {
    double** arr = new double*[row];
    for (int i = 0; i < row; i++) {
        arr[i] = new double[col];
    }
    return arr;
}

void free2DArray(double** arr, const int row) {
    for (int i = 0; i < row; i++) {
        delete[] arr[i];
    }
    delete[] arr;
}

void print_progress_bar(
    const int current_epoch, const int num_epochs, const int current_batch,
    const int num_batches, const double current_loss,
    const decltype(std::chrono::high_resolution_clock::now()) epoch_start_time,
    const double mean_loss) {
    const int bar_width = 50;
    const double curr_progress = (double)current_batch / num_batches;
    const int pos = bar_width * curr_progress;

    printf("\rEpoch %d/%d [", current_epoch, num_epochs);
    for (int i = 0; i < bar_width; i++) {
        if (i < pos)
            printf("=");
        else if (i == pos)
            printf(">");
        else
            printf(" ");
    }
    auto current_time = std::chrono::high_resolution_clock::now();
    printf(
        "] Batch %d/%d | Time: %fs | Last Batch Loss: %f", current_batch,
        num_batches,
        std::chrono::duration<double>(current_time - epoch_start_time).count(),
        current_loss);
    fflush(stdout);
}

// copy 2D array content from host to device
// dest_d: 2D array on the device (arr[i] on device)
// src_h: 2D array on the host
// row, col: dimensions of the 2D array
void copy2DArrayToDevice(double** dest_d, double** src_h, const int row,
                         const int col) {
    double** arr = new double*[row];
    cudaMemcpy(arr, dest_d, row * sizeof(double*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < row; i++) {
        cudaMemcpyAsync(arr[i], src_h[i], col * sizeof(double),
                        cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();

    delete[] arr;
}

// copy 2D array content from device to host
// dest_h: 2D array on the host
// src_d: 2D array on the device (arr[i] on device)
// row, col: dimensions of the 2D array
void copy2DArrayFromDevice(double** dest_h, double** src_d, const int row,
                           const int col) {
    double** arr = new double*[row];
    cudaMemcpy(arr, src_d, row * sizeof(double*), cudaMemcpyDeviceToHost);

    for (int i = 0; i < row; i++) {
        cudaMemcpyAsync(dest_h[i], arr[i], col * sizeof(double),
                        cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();

    delete[] arr;
}

// return a 2D array of shape (row, col) on the device
// copy_src: host 2d arr with same shape. if not nullptr, copy the content of
// copy_src to the new array
double** create2DArrayCuda(const int row, const int col,
                           double** const copy_src) {
    // Note: can be re-written in recursion form to support higher dimensions

    double** arr_1d = new double*[row];
    for (int i = 0; i < row; i++) {
        cudaMallocAsync(&arr_1d[i], col * sizeof(double), 0);
    }

    // all elements in arr_1d are ptrs on the device, but arr_1d itself is on
    // the host
    // => allocate another 2D array on device to store the ptrs

    double** arr_2d = nullptr;
    cudaMallocAsync(&arr_2d, row * sizeof(double*), 0);

    // copy the ptrs from arr_1d to arr_2d
    cudaMemcpyAsync(arr_2d, arr_1d, row * sizeof(double*),
                    cudaMemcpyHostToDevice);

    if (copy_src) {
        for (int i = 0; i < row; i++) {
            cudaMemcpyAsync(arr_1d[i], copy_src[i], col * sizeof(double),
                            cudaMemcpyHostToDevice);
        }
    }

    // wait for the copy to finish
    cudaDeviceSynchronize();

    delete[] arr_1d;

    return arr_2d;
}

double*** create3DArrayCuda(const int row, const int col, const int depth) {
    double*** temp_3d_arr = new double**[row];
    for (int i = 0; i < row; i++)
        temp_3d_arr[i] = create2DArrayCuda(col, depth);

    double*** cuda_3d_arr;
    cudaMallocAsync(&cuda_3d_arr, row * sizeof(double**), 0);
    cudaMemcpyAsync(cuda_3d_arr, temp_3d_arr, row * sizeof(double**),
                    cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    delete[] temp_3d_arr;

    return cuda_3d_arr;
}

// free the memory of the 2D array on the device
// copy_image: host 2d arr with same shape. if not nullptr, copy the content of
// the device array to copy_image
void free2DArrayCuda(double** arr, const int row, double** const copy_image,
                     const int col) {
    double** arr_host = new double*[row];
    cudaMemcpy(arr_host, arr, row * sizeof(double*), cudaMemcpyDeviceToHost);

    if (copy_image) {
        for (int i = 0; i < row; i++) {
            cudaMemcpy(copy_image[i], arr_host[i], col * sizeof(double),
                       cudaMemcpyDeviceToHost);
        }
    }

    for (int i = 0; i < row; i++) {
        cudaFree(arr_host[i]);
    }

    delete[] arr_host;
    cudaFree(arr);
}

void free3DArrayCuda(double*** arr, const int row, const int col) {
    double*** arr_host = new double**[row];
    cudaMemcpy(arr_host, arr, row * sizeof(double**), cudaMemcpyDeviceToHost);

    for (int i = 0; i < row; i++) free2DArrayCuda(arr_host[i], col);

    delete[] arr_host;
    cudaFree(arr);
}

__global__ void findMax(double* const max_val, double** const in,
                        const int len) {
    extern __shared__ double s_arr[];

    const int thread_id = threadIdx.x;
    const int i = blockIdx.x;

    s_arr[thread_id] = -INFINITY;
    for (int j = thread_id; j < len; j += blockDim.x) {
        if (in[i][j] > s_arr[thread_id]) {
            s_arr[thread_id] = in[i][j];
        }
    }
    __syncthreads();

    // iteratively move the max value to the first element
    for (int remaining_threads = blockDim.x / 2; remaining_threads > 0;
         remaining_threads /= 2) {
        if (thread_id < remaining_threads &&
            s_arr[thread_id] < s_arr[thread_id + remaining_threads]) {
            s_arr[thread_id] = s_arr[thread_id + remaining_threads];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        max_val[i] = s_arr[0];
    }
}

__global__ void computeRawSoftmax(double** out, double** in, double* max_val,
                                  double* sum_exp, int len) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double s_partial_sums[];

    s_partial_sums[threadIdx.x] = 0.0;
    if (j < len) {
        out[i][j] = exp(in[i][j] - max_val[i]);
        s_partial_sums[threadIdx.x] = out[i][j];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            s_partial_sums[threadIdx.x] += s_partial_sums[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_exp[i] = s_partial_sums[0];
    }
}

__global__ void normalizeSoftmax(double** out, double* sum_exp, int len) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < len) {
        out[i][j] /= sum_exp[i];
    }
}

void softmaxCuda(double** out, double** in, int len, int batch_size) {
    // "in", "out" are device ptr
    int num_threads = 256;
    int num_blocks = DIV_CEIL(len, num_threads);

    double* max_val = nullptr;
    cudaMalloc(&max_val, batch_size * sizeof(double));

    findMax<<<batch_size, num_threads, num_threads * sizeof(double)>>>(max_val,
                                                                       in, len);

    double* sum_exp = nullptr;
    cudaMalloc(&sum_exp, batch_size * sizeof(double));
    cudaMemset(sum_exp, 0, batch_size * sizeof(double));

    computeRawSoftmax<<<dim3(num_blocks, batch_size), num_threads,
                        num_threads * sizeof(double)>>>(out, in, max_val,
                                                        sum_exp, len);

    normalizeSoftmax<<<dim3(num_blocks, batch_size), num_threads>>>(
        out, sum_exp, len);

    cudaFree(max_val);
    cudaFree(sum_exp);
}

/*
 * CUDA does not natively support atomic addition for doubles.
 * This function is adapted from the following source:
 * https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
 */
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}