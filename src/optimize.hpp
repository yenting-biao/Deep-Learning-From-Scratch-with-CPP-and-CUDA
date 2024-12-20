/**
 * @file optimize.hpp
 * @brief Provides supporting code for the optimization step.
 */

#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

#include <cuda.h>
#include <cuda_runtime.h>

struct HostLayerOptimizationPointers {
    double* grad_act = nullptr;
    double* delta = nullptr;
    double** w_next_transpose = nullptr;
    double** temp = nullptr;
    double*** delta_2D = nullptr;
    double** grad_w = nullptr;
    double* grad_b = nullptr;
    double** temp_grad_w = nullptr;
    double** input2D = nullptr;
};

struct DeviceLayerOptimizationPointers {
    double** grad_act = nullptr;
    double** delta = nullptr;
    double** w_next_transpose = nullptr;
    double*** temp = nullptr;
    double*** delta_2D = nullptr;
    double** grad_w = nullptr;
    double* grad_b = nullptr;
    double*** temp_grad_w = nullptr;
    double*** input2D = nullptr;
};

void _host_allocateMemoryForLayerOptimizationPointers(
    HostLayerOptimizationPointers& layer_optimization_pointers,
    int max_output_size, int max_input_size, int batch_size);
void _device_allocateMemoryForLayerOptimizationPointers(
    DeviceLayerOptimizationPointers& layer_optimization_pointers,
    int max_output_size, int max_input_size, int batch_size);
void _host_freeMemoryForLayerOptimizationPointers(
    HostLayerOptimizationPointers& layer_optimization_pointers,
    int max_output_size, int batch_size);
void _device_freeMemoryForLayerOptimizationPointers(
    DeviceLayerOptimizationPointers& layer_optimization_pointers,
    int max_output_size, int batch_size);

__global__ void device_multiplyArrays(double** device_delta,
                                      double** device_grad_act, int batch_size,
                                      int size);

__global__ void device_multiplyArrays(double** device_delta, double*** temp,
                                      double** device_grad_act, int batch_size,
                                      int size);

__global__ void device_transposeMatrix(double** device_transposed_matrix,
                                       double** device_source_matrix,
                                       int rows_of_source_matrix,
                                       int cols_of_source_matrix);

__global__ void device_multiplyMatrix(double*** out, double** a, double*** b,
                                      int row_a, int col_a, int row_b,
                                      int col_b, int batch_size);

__global__ void device_multiplyMatrix(double*** out, double*** a, double*** b,
                                      int row_a, int col_a, int row_b,
                                      int col_b, int batch_size);

__global__ void device_updateDelta2DAndGradB(double*** device_delta_2D,
                                             double** device_delta,
                                             double* device_grad_b,
                                             int batch_size, int size);

__global__ void device_setInput2D(double*** device_input2D,
                                  double** device_input, int batch_size,
                                  int size);

__global__ void device_clipGrad(double*** device_grad, int batch_size, int row,
                                int col);

__global__ void device_updateGradW(double** device_grad_w,
                                   double*** device_temp_grad_w, int batch_size,
                                   int row, int col);

__global__ void device_updateWeightsAndBiases(
    double** device_w, double** last_v_w, double** device_grad_w,
    double* device_b, double* last_v_b, double* device_grad_b, double momentum,
    double lr, int row, int col, int batch_size);

#endif  // OPTIMIZE_HPP