#include "optimize.hpp"
#include "utils.hpp"

void _host_allocateMemoryForLayerOptimizationPointers(
    HostLayerOptimizationPointers& layer_optimization_pointers,
    const int max_output_size, const int max_input_size, const int batch_size) {
    auto& [grad_act, delta, w_next_transpose, temp, delta_2D, grad_w, grad_b,
           temp_grad_w, input2D] = layer_optimization_pointers;
    grad_act = new double[max_output_size];
    delta = new double[max_output_size];
    w_next_transpose = new double*[max_output_size];
    temp = new double*[max_output_size];
    delta_2D = new double**[batch_size];
    for (int i = 0; i < batch_size; i++) {
        delta_2D[i] = new double*[max_output_size];
        for (int j = 0; j < max_output_size; j++)
            delta_2D[i][j] = new double[1];
    }
    grad_w = new double*[max_output_size];
    grad_b = new double[max_output_size]();
    temp_grad_w = new double*[max_output_size];
    input2D = new double*[1];
    input2D[0] = new double[max_input_size];
    for (int i = 0; i < max_output_size; i++) {
        w_next_transpose[i] = new double[max_output_size];
        temp[i] = new double[1];
        grad_w[i] = new double[max_input_size]();
        temp_grad_w[i] = new double[max_input_size];
    }
}

void _device_allocateMemoryForLayerOptimizationPointers(
    DeviceLayerOptimizationPointers& layer_optimization_pointers,
    const int max_output_size, const int max_input_size, const int batch_size) {
    auto& [grad_act, delta, w_next_transpose, temp, delta_2D, grad_w, grad_b,
           temp_grad_w, input2D] = layer_optimization_pointers;
    grad_act = create2DArrayCuda(batch_size, max_output_size);
    delta = create2DArrayCuda(batch_size, max_output_size);
    w_next_transpose = create2DArrayCuda(max_output_size, max_output_size);
    temp = create3DArrayCuda(batch_size, max_output_size, 1);
    delta_2D = create3DArrayCuda(batch_size, max_output_size, 1);
    grad_w = create2DArrayCuda(max_output_size, max_input_size);
    cudaMallocAsync(&grad_b, max_output_size * sizeof(double), 0);
    temp_grad_w =
        create3DArrayCuda(batch_size, max_output_size, max_input_size);
    input2D = create3DArrayCuda(batch_size, 1, max_input_size);
    cudaDeviceSynchronize();
}

void _host_freeMemoryForLayerOptimizationPointers(
    HostLayerOptimizationPointers& layer_optimization_pointers,
    const int max_output_size, const int batch_size) {
    auto [grad_act, delta, w_next_transpose, temp, delta_2D, grad_w, grad_b,
          temp_grad_w, input2D] = layer_optimization_pointers;
    delete[] grad_act;
    delete[] delta;
    free2DArray(w_next_transpose, max_output_size);
    free2DArray(temp, max_output_size);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < max_output_size; j++) delete[] delta_2D[i][j];
        delete[] delta_2D[i];
    }
    delete[] delta_2D;
    free2DArray(grad_w, max_output_size);
    delete[] grad_b;
    free2DArray(temp_grad_w, max_output_size);
    free2DArray(input2D, 1);
}

void _device_freeMemoryForLayerOptimizationPointers(
    DeviceLayerOptimizationPointers& layer_optimization_pointers,
    const int max_output_size, const int batch_size) {
    auto [grad_act, delta, w_next_transpose, temp, delta_2D, grad_w, grad_b,
          temp_grad_w, input2D] = layer_optimization_pointers;
    free2DArrayCuda(grad_act, batch_size);
    free2DArrayCuda(delta, batch_size);
    free2DArrayCuda(w_next_transpose, max_output_size);
    free3DArrayCuda(temp, batch_size, max_output_size);
    free3DArrayCuda(delta_2D, batch_size, max_output_size);
    free2DArrayCuda(grad_w, max_output_size);
    cudaFree(grad_b);
    free3DArrayCuda(temp_grad_w, batch_size, max_output_size);
    free3DArrayCuda(input2D, batch_size, 1);
}

__global__ void device_multiplyArrays(double** const device_delta,
                                      double** const device_grad_act,
                                      const int batch_size, const int size) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int curr_data = thread_id / size;
    const int i = thread_id % size;
    if (curr_data < batch_size && i < size) {
        device_delta[curr_data][i] *= device_grad_act[curr_data][i];
    }
}

__global__ void device_multiplyArrays(double** const device_delta,
                                      double*** const temp,
                                      double** const device_grad_act,
                                      const int batch_size, const int size) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int curr_data = thread_id / size;
    const int i = thread_id % size;
    if (curr_data < batch_size && i < size) {
        device_delta[curr_data][i] =
            temp[curr_data][i][0] * device_grad_act[curr_data][i];
    }
}

__global__ void device_transposeMatrix(double** const device_transposed_matrix,
                                       double** const device_source_matrix,
                                       const int rows_of_source_matrix,
                                       const int cols_of_source_matrix) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_id_of_source_matrix = thread_id / cols_of_source_matrix;
    const int col_id_of_source_matrix = thread_id % cols_of_source_matrix;
    if (row_id_of_source_matrix < rows_of_source_matrix &&
        col_id_of_source_matrix < cols_of_source_matrix) {
        device_transposed_matrix
            [col_id_of_source_matrix][row_id_of_source_matrix] =
                device_source_matrix[row_id_of_source_matrix]
                                    [col_id_of_source_matrix];
    }
}

__global__ void device_multiplyMatrix(
    double*** const out,
    double** const a,  // the same matrix is used for every data, hence it is 2D
    double*** const b, const int row_a, const int col_a, const int row_b,
    const int col_b, const int batch_size) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int curr_data = thread_id / (row_a * col_b);
    const int row_index = (thread_id % (row_a * col_b)) / col_b;
    const int col_index = (thread_id % (row_a * col_b)) % col_b;

    if (curr_data < batch_size && row_index < row_a && col_index < col_b) {
        double temp_sum = 0.0;
        for (int k = 0; k < col_a; k++) {
            temp_sum += a[row_index][k] * b[curr_data][k][col_index];
        }
        out[curr_data][row_index][col_index] = temp_sum;
    }
}

__global__ void device_multiplyMatrix(
    double*** const out,
    double*** const a,  // each data uses a different matrix; curr_data uses
                        // matrix a[curr_data]
    double*** const b, const int row_a, const int col_a, const int row_b,
    const int col_b, const int batch_size) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int curr_data = thread_id / (row_a * col_b);
    const int row_index = (thread_id % (row_a * col_b)) / col_b;
    const int col_index = (thread_id % (row_a * col_b)) % col_b;

    if (curr_data < batch_size && row_index < row_a && col_index < col_b) {
        double temp_sum = 0.0;
        for (int k = 0; k < col_a; k++) {
            temp_sum += a[curr_data][row_index][k] * b[curr_data][k][col_index];
        }
        out[curr_data][row_index][col_index] = temp_sum;
    }
}

__global__ void device_updateDelta2DAndGradB(double*** const device_delta_2D,
                                             double** const device_delta,
                                             double* const device_grad_b,
                                             const int batch_size,
                                             const int size) {
    const int index = blockIdx.x;  // Each block handles one index
    if (index >= size) return;

    extern __shared__ double
        s_partial_sums[];  // One element per thread to store its partial sum
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;
    double my_partial_sum = 0.0;

    /*
     * Step 1: Each thread iterates over its set of data, setting delta_2D
     *         and keeping a running sum in the process. The sum will be used
     *         to update device_grad_b[i] later.
     */
    const int stride = num_threads;
    for (int curr_data = thread_id; curr_data < batch_size;
         curr_data += stride) {
        const double curr_delta = device_delta[curr_data][index];
        device_delta_2D[curr_data][index][0] = curr_delta;
        my_partial_sum += curr_delta;
    }
    s_partial_sums[thread_id] = my_partial_sum;
    __syncthreads();

    /*
     * Step 2: Sum up all the partial sums into one.
     * This is done in log2(num_threads) steps, where half of the remaining
     * threads add the other half's partial sum to their own.
     */
    for (int threads_remaining = num_threads / 2; threads_remaining > 0;
         threads_remaining /= 2) {
        if (thread_id < threads_remaining) {
            s_partial_sums[thread_id] +=
                s_partial_sums[thread_id + threads_remaining];
        }
        __syncthreads();
    }

    /*
     * Step 3: One thread updates device_grad_b[i]. Because of the first two
     * steps, the number of atomic adds is reduced.
     */
    if (thread_id == 0) {
        atomicAddDouble(&device_grad_b[index], s_partial_sums[0]);
    }
}

__global__ void device_setInput2D(double*** const device_input2D,
                                  double** const device_input,
                                  const int batch_size, const int size) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int curr_data = thread_id / size;
    const int i = thread_id % size;

    if (curr_data < batch_size && i < size) {
        device_input2D[curr_data][0][i] = device_input[curr_data][i];
    }
}

__global__ void device_clipGrad(double*** const device_grad,
                                const int batch_size, const int row,
                                const int col) {
    const int curr_data = blockIdx.x;  // Each block handles one data
    if (curr_data >= batch_size) return;

    extern __shared__ double
        s_partial_sums[];  // One element per thread to store its partial sum
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total_elements = row * col;
    double my_partial_sum = 0.0;

    /*
     * Step 1: Each thread computes a partial sum of squares of its gradient
     *         elements.
     */
    const int stride = num_threads;
    for (int i = thread_id; i < total_elements; i += stride) {
        const int row_index = i / col;
        const int col_index = i % col;
        const double curr_grad = device_grad[curr_data][row_index][col_index];
        my_partial_sum += curr_grad * curr_grad;
    }
    s_partial_sums[thread_id] = my_partial_sum;
    __syncthreads();

    /*
     * Step 2: Sum up all the partial sums into one.
     * This is done in log2(num_threads) steps, where half of the remaining
     * threads add the other half's partial sum to their own.
     */
    for (int threads_remaining = num_threads / 2; threads_remaining > 0;
         threads_remaining /= 2) {
        if (thread_id < threads_remaining) {
            s_partial_sums[thread_id] +=
                s_partial_sums[thread_id + threads_remaining];
        }
        __syncthreads();
    }

    /*
     * Step 3: One thread computes the scaling factor for everyone in the block.
     */
    double scale = 1.0;
    if (thread_id == 0) {
        double norm = sqrt(s_partial_sums[0]);
        double max_norm = 1.0;
        if (norm > max_norm) {
            scale = max_norm / norm;
        }
        s_partial_sums[0] = scale;
    }
    __syncthreads();

    /*
     * Step 4: Scale the gradients if necessary.
     */
    scale = s_partial_sums[0];
    if (scale < 1.0) {
        for (int i = thread_id; i < total_elements; i += stride) {
            const int row_index = i / col;
            const int col_index = i % col;
            device_grad[curr_data][row_index][col_index] *= scale;
        }
    }
}

__global__ void device_updateGradW(double** const device_grad_w,
                                   double*** const device_temp_grad_w,
                                   const int batch_size, const int row,
                                   const int col) {
    /*
     * Each thread is responsible for one element in the gradient matrix.
     */
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_index = thread_id / col;
    const int col_index = thread_id % col;

    if (row_index < row && col_index < col) {
        double temp_sum = 0.0;
        for (int i = 0; i < batch_size; i++) {
            temp_sum += device_temp_grad_w[i][row_index][col_index];
        }
        device_grad_w[row_index][col_index] = temp_sum;
    }
}

__global__ void device_updateWeightsAndBiases(
    double** const device_w, double** const last_v_w,
    double** const device_grad_w, double* const device_b,
    double* const last_v_b, double* const device_grad_b, double momentum,
    const double lr, const int row, const int col, const int batch_size) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_index = thread_id / col;
    const int col_index = thread_id % col;

    if (row_index < row && col_index < col) {
        double v = momentum * last_v_w[row_index][col_index] -
                   lr * device_grad_w[row_index][col_index] / batch_size;
        device_w[row_index][col_index] += v;
        last_v_w[row_index][col_index] = v;
        if (col_index == 0) {
            v = momentum * last_v_b[row_index] -
                lr * device_grad_b[row_index] / batch_size;
            device_b[row_index] += v;
            last_v_b[row_index] = v;
        }
    }
}