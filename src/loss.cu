#include <stdio.h>

#include "loss.hpp"
#include "utils.hpp"

double MSELoss::compute(const double* const ground_truth,
                        const double* const prediction, const int n) {
    // Implementation of the mean squared error loss function

    // ground_truth, prediction: 1D array of size n
    double error_sum = 0.0;

    for (int i = 0; i < n; i++) {
        const double diff = ground_truth[i] - prediction[i];
        error_sum += diff * diff;
    }

    return (double)(error_sum / n);
}

/*
 * Uses the technique of using half of the threads to do reduction.
 */
__global__ void computeLoss_mse(double* const loss, double** const ground_truth,
                                double** const prediction, const int n,
                                const int batch_size) {
    extern __shared__ double s_partial_sums[];
    const int thread_id_in_grid = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id_in_block = threadIdx.x;
    const int num_threads = blockDim.x;
    const int curr_data = thread_id_in_grid / n;
    const int curr_data_output_idx = thread_id_in_grid % n;

    s_partial_sums[thread_id_in_block] = 0.0;
    if (curr_data < batch_size && curr_data_output_idx < n) {
        const double diff = prediction[curr_data][curr_data_output_idx] -
                            ground_truth[curr_data][curr_data_output_idx];
        s_partial_sums[thread_id_in_block] = diff * diff;
    }
    __syncthreads();

    // assumption: num_threads per block is a power of 2
    for (int remaining_threads = num_threads / 2; remaining_threads > 0;
         remaining_threads /= 2) {
        if (thread_id_in_block < remaining_threads) {
            s_partial_sums[thread_id_in_block] +=
                s_partial_sums[thread_id_in_block + remaining_threads];
        }
        __syncthreads();
    }

    if (thread_id_in_block == 0) {
        atomicAddDouble(loss, s_partial_sums[0]);
    }
}

double MSELoss::compute_OnDev(double** device_ground_truth,
                              double** device_prediction,
                              const int n,  // array size (2nd dimension)
                              const int batch_size  // 1st dimension
) {
    double* device_loss;
    cudaMalloc(&device_loss, sizeof(double));
    cudaMemset(device_loss, 0, sizeof(double));

    computeLoss_mse<<<(batch_size * n + 256 - 1) / 256, 256,
                      256 * sizeof(double)>>>(device_loss, device_ground_truth,
                                              device_prediction, n, batch_size);

    double host_loss;
    cudaMemcpy(&host_loss, device_loss, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(device_loss);

    return host_loss / n;
}

__global__ static void _MSELoss_device_gradient(double** grad,
                                                double** ground_truth,
                                                double** prediction,
                                                const int batch_size,
                                                const int n) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int curr_data = thread_id / n;
    const int i = thread_id % n;

    if (curr_data < batch_size && i < n) {
        grad[curr_data][i] =
            2 * (prediction[curr_data][i] - ground_truth[curr_data][i]) / n;
    }
}

void MSELoss::gradient(double* const grad, const double* const ground_truth,
                       const double* const prediction, const int n,
                       const bool use_cuda, const int batch_size,
                       double** const device_delta,
                       double** const device_ground_truth,
                       double** const out_act) {
    // Implementation of the gradient of the mean squared error loss function

    // ground_truth, prediction: 1D array of size n

    if (!use_cuda) {
        for (int i = 0; i < n; i++) {
            grad[i] = 2 * (prediction[i] - ground_truth[i]) / n;
        }
    } else {
        _MSELoss_device_gradient<<<(batch_size * n + 256 - 1) / 256, 256>>>(
            device_delta, device_ground_truth, out_act, batch_size, n);
    }
}

double CrossEntropyLoss::compute(const double* const ground_truth,
                                 const double* const prediction, const int n) {
    // Implementation of the cross entropy loss function with softmax applied
    // first ground_truth: probability distributions prediction: raw scores
    // (logits)

    double entropy = 0.0;
    double* softmax_pred = new double[n];
    softmax(softmax_pred, prediction, n);

    for (int i = 0; i < n; i++) {
        entropy += -ground_truth[i] * log(softmax_pred[i]);
    }

    delete[] softmax_pred;
    return entropy;
}

__global__ void computeLoss_ce(double* const loss, double** const ground_truth,
                               double** const prediction, const int n,
                               const int batch_size) {
    extern __shared__ double s_partial_sums[];

    // block.y: batch index => i
    // block.x: data index => j
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    s_partial_sums[threadIdx.x] = 0.0;
    if (j < n) {
        s_partial_sums[threadIdx.x] =
            -ground_truth[i][j] * log(prediction[i][j]);
    }
    __syncthreads();

    // assumption: blockDim.x is a power of 2
    for (int remaining_threads = blockDim.x / 2; remaining_threads > 0;
         remaining_threads /= 2) {
        if (threadIdx.x < remaining_threads) {
            s_partial_sums[threadIdx.x] +=
                s_partial_sums[threadIdx.x + remaining_threads];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAddDouble(loss, s_partial_sums[0]);
    }
}

double CrossEntropyLoss::compute_OnDev(
    double** device_ground_truth, double** device_prediction,
    const int n,          // array size (2nd dimension)
    const int batch_size  // 1st dimension
) {
    double* device_loss;
    cudaMalloc(&device_loss, sizeof(double));
    cudaMemset(device_loss, 0, sizeof(double));

    int num_threads = 256;

    double** device_softmax_pred = create2DArrayCuda(batch_size, n);

    softmaxCuda(device_softmax_pred, device_prediction, n, batch_size);

    computeLoss_ce<<<dim3(DIV_CEIL(n, num_threads), batch_size), num_threads,
                     num_threads * sizeof(double)>>>(
        device_loss, device_ground_truth, device_softmax_pred, n, batch_size);

    double host_loss;
    cudaMemcpy(&host_loss, device_loss, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_loss);
    free2DArrayCuda(device_softmax_pred, batch_size);

    return host_loss;
}

void CrossEntropyLoss::gradient(
    double* const grad, const double* const ground_truth,
    const double* const prediction, const int n, bool use_cuda, int batch_size,
    double** device_delta, double** device_ground_truth, double** out_act) {
    // Implementation of the gradient of the cross entropy loss function
    // Note: under the scene, ground_truth and prediction are prob.
    // distributions

    double* softmax_pred = new double[n];
    softmax(softmax_pred, prediction, n);

    for (int i = 0; i < n; i++) {
        grad[i] = softmax_pred[i] - ground_truth[i];
    }

    delete[] softmax_pred;
}