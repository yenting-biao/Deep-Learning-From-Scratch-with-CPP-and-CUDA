#include <stdlib.h>

#include <cmath>
#include <iostream>

#include "cuda_error_handler.hpp"
#include "layer.hpp"
#include "optimize.hpp"
#include "utils.hpp"

#define MULT_32(x) \
    ((((x) + 31) / 32) * 32)  // rounds up to nearest multiple of 32
#define MIN(x, y) ((x) < (y) ? (x) : (y))

void clip_grad(double** grad, int row, int col) {
    double max_norm = 1.0;
    double norm = 0.0;

    // Calculate the L2 norm of the gradients
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            norm += grad[i][j] * grad[i][j];
        }
    }
    norm = sqrt(norm);

    // If the norm exceeds the maximum allowed, scale the gradients
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                grad[i][j] *= scale;
            }
        }
    }
}

class Activation {
   public:
    static inline double none(const double x) { return x; }

    static inline double relu(const double x) { return (x > 0) ? x : 0; }

    static inline double sigmoid(const double x) { return 1 / (1 + exp(-x)); }

    // initialization of the func_arr need to be outside the class
    static double (*func_vec[])(double);
};

// array of function pointers, access the function by using Layer.act as the
// index
double (*Activation::func_vec[])(double) = {Activation::none, Activation::relu,
                                            Activation::sigmoid};

__device__ double none_dev(const double x) { return x; }

__device__ double relu_dev(const double x) { return (x > 0) ? x : 0; }

__device__ double sigmoid_dev(const double x) { return 1 / (1 + exp(-x)); }

__device__ double (*func_vec_dev[])(double) = {none_dev, relu_dev, sigmoid_dev};

__global__ void activateCuda(double** const out, double** const in,
                             const int row, const int col, const int act) {
    const int i = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < row && j < col) {
        out[i][j] = func_vec_dev[act](in[i][j]);
    }
}

class Activation_grad {
   public:
    static inline __host__ __device__ double none(const double x) {
        return 1.0;
    }

    static inline __host__ __device__ double relu(const double x) {
        return (x > 0) ? 1.0 : 0.0;
    }

    static inline __host__ __device__ double sigmoid(const double x) {
        return x * (1 - x);
    }

    // initialization of the func_arr need to be outside the class
    static double (*func_vec[])(double);
};

// array of function pointers, access the function by using Layer.act as the
// index
__device__ double (*act_grad_func_vec[])(double) = {
    Activation_grad::none, Activation_grad::relu, Activation_grad::sigmoid};
double (*Activation_grad::func_vec[])(double) = {
    Activation_grad::none, Activation_grad::relu, Activation_grad::sigmoid};

Layer::Layer(const unsigned long long input_size,
             const unsigned long long output_size, const int activation,
             const int batch_size, bool use_cuda, double** const weights,
             double* const biases)
    : input_size(input_size),
      output_size(output_size),
      batch_size(batch_size),
      act(activation),
      use_cuda(use_cuda) {
    if (use_cuda) {
        out_raw = create2DArrayCuda(batch_size, output_size);
        out_act = out_raw;
        if (act != ACT_NONE) {
            out_act = create2DArrayCuda(batch_size, output_size);
        }

        last_v_w = create2DArrayCuda(output_size, input_size);
        cudaMalloc(&last_v_b, output_size * sizeof(double));
        cudaMemset(last_v_b, 0, output_size * sizeof(double));
        double** h_last_v_w = new double*[output_size];
        CUDA_CHECK_ERROR(cudaMemcpy(h_last_v_w, last_v_w,
                                    output_size * sizeof(double*),
                                    cudaMemcpyDeviceToHost));
        for (int i = 0; i < output_size; i++) {
            CUDA_CHECK_ERROR(
                cudaMemset(h_last_v_w[i], 0, input_size * sizeof(double)));
        }
        delete[] h_last_v_w;
    } else {
        // allocate memory for out_raw and out_act (output_size)
        out_raw = new double*[batch_size];
        for (int i = 0; i < batch_size; i++) {
            out_raw[i] = new double[output_size];
        }

        // optimize the memory allocation for out_act
        if (act == ACT_NONE) {
            out_act = out_raw;
        } else {
            out_act = new double*[batch_size];
            for (int i = 0; i < batch_size; i++) {
                out_act[i] = new double[output_size];
            }
        }

        // allocate memory for last_v_w and last_v_b
        last_v_w = new double*[output_size];
        for (int i = 0; i < output_size; i++) {
            last_v_w[i] = new double[input_size]();
        }

        last_v_b = new double[output_size]();
    }

    // allocate memory for w and b
    w = new double*[output_size];
    for (int i = 0; i < output_size; i++) {
        w[i] = new double[input_size];
    }

    if (weights != nullptr) {
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                w[i][j] = weights[i][j];
            }
        }
    } else {
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                w[i][j] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
    }

    b = new double[output_size];

    if (biases != nullptr) {
        for (int i = 0; i < output_size; i++) {
            b[i] = biases[i];
        }
    } else {
        for (int i = 0; i < output_size; i++) {
            b[i] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    if (use_cuda) {
        // copy data to the device
        double** w_cuda = create2DArrayCuda(output_size, input_size, w);

        double* b_cuda = nullptr;
        cudaMalloc(&b_cuda, output_size * sizeof(double));
        cudaMemcpy(b_cuda, b, output_size * sizeof(double),
                   cudaMemcpyHostToDevice);

        // free the memory on the host
        for (int i = 0; i < output_size; i++) {
            delete[] w[i];
        }
        delete[] w;
        delete[] b;

        w = w_cuda;
        b = b_cuda;
    }
}

Layer::~Layer() {
    if (use_cuda) {
        if (act != ACT_NONE) {
            free2DArrayCuda(out_act, batch_size);
        }
        free2DArrayCuda(out_raw, batch_size);
        free2DArrayCuda(last_v_w, output_size);
        cudaFree(last_v_b);

        free2DArrayCuda(w, output_size);
        cudaFree(b);
    } else {
        free2DArray(out_raw, batch_size);

        if (act != ACT_NONE) {
            free2DArray(out_act, batch_size);
        }

        // free w and b
        free2DArray(w, output_size);
        delete[] b;

        // free last_v_w and last_v_b
        free2DArray(last_v_w, output_size);
        delete[] last_v_b;
    }
}

// do (Wx + b) on the device
// out, W, x, b: 2D arrays on the device
// PS: row_out = batch size, col_out = output size
// row_W, col_W: dimensions of W (== (output size, input size))
// row_x, col_x: dimensions of x (== (batch size, input_size))
// b: 1D array on the device, length = col_W
__global__ void linearForwardCuda(double** const out, double** const W,
                                  double** const x, double* const b,
                                  const int row_out, const int col_out,
                                  const int middle) {
    const int i = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < row_out && j < col_out) {
        out[i][j] = 0.0;

        for (int k = 0; k < middle; k++) {
            out[i][j] += W[j][k] * x[i][k];
        }

        out[i][j] += b[j];
    }
}

void Layer::forward(double** const input, int batch_size,
                    double** const output) {
    // Implement the forward pass of the layer for batch processing
    // output: 2D array of shape (batch_size, output_size)
    // input: 2D array of shape (batch_size, input_size)

    // Note that w.shape = (output_size, input_size), input.shape = (batch_size,
    // input_size) thus call multiplyMatrixBTCuda()
    if (use_cuda) {
        int num_thread = 256;

        linearForwardCuda<<<dim3(DIV_CEIL(output_size, num_thread), batch_size),
                            num_thread>>>(out_raw, w, input, b, batch_size,
                                          output_size, input_size);

        if (act != ACT_NONE) {
            activateCuda<<<dim3(DIV_CEIL(output_size, num_thread), batch_size),
                           num_thread>>>(out_act, out_raw, batch_size,
                                         output_size, act);
        }

        // copy the data back to the host
        if (output != nullptr) {
            copy2DArrayFromDevice(output, out_act, batch_size, output_size);
        }
    } else {
        for (int curr_data = 0; curr_data < batch_size; curr_data++) {
            // calculate the raw output
            for (int i = 0; i < output_size; i++) {
                out_raw[curr_data][i] = 0.0;

                for (int j = 0; j < input_size; j++) {
                    out_raw[curr_data][i] += w[i][j] * input[curr_data][j];
                }

                out_raw[curr_data][i] += b[i];
            }

            // activate the raw output
            if (act != ACT_NONE) {  // if no activation function is used,
                                    // out_act is the same as out_raw
                for (int i = 0; i < output_size; i++) {
                    out_act[curr_data][i] =
                        Activation::func_vec[act](out_raw[curr_data][i]);
                }
            }

            // copy the activated output to the output array
            if (output != nullptr) {
                for (int i = 0; i < output_size; i++) {
                    output[curr_data][i] = out_act[curr_data][i];
                }
            }
        }
    }
}

/**
 * @brief Optimizes weights of one layer using backpropagation for one batch.
 *
 * Definitions:
 * Previous layer: layer_index - 1
 * Current layer: layer_index
 * Next layer: layer_index + 1
 *
 * Backpropagation essentially involves calling this function for each layer,
 * starting from the last layer and moving towards the first layer, for a total
 * of nLayers times. That is, the weights of the entire model after updated
 * layer by layer, beginning with the last layer. At each layer, the gradients
 * for the weights and biases of the layer are calculated using the gradients
 * from the next layer (e.g., layer 0 uses the gradients from layer 1). The
 * gradients are then used to update the weights and biases of the layer.
 *
 * @param layer_index Index of the current layer in the model.
 * @param nLayers Total number of layers in the model.
 * @param delta_next Gradients from the next layer.
 * @param loss_func Pointer to the loss function used for calculating gradients.
 * @param ground_truth Ground truth labels for the batch.
 * @param w_next Weights of the next layer.
 * @param next_layer_output_size Output size of the next layer.
 * @param input Input data for the current layer (either raw input or output
 * from previous layer).
 * @param lr Learning rate for the optimization.
 * @param batch_size Number of data samples in the batch.
 * @return Gradients of the weights of current layer with respect to the input
 * of the current layer.
 */
void Layer::host_optimize(
    const HostLayerOptimizationPointers& layer_optimization_pointers,
    const int layer_index, const int nLayers, LossFunction* const loss_func,
    double** const ground_truth, double** const w_next,
    const int next_layer_output_size, double** const input, const double lr,
    const int batch_size, const double momentum) {
    auto [grad_act, delta, w_next_transpose, temp, delta_2D, grad_w, grad_b,
          temp_grad_w, input2D] = layer_optimization_pointers;
    memset(grad_b, 0, output_size * sizeof(double));
    for (int i = 0; i < output_size; i++)
        memset(grad_w[i], 0, input_size * sizeof(double));

    for (int curr_data = 0; curr_data < batch_size; curr_data++) {
        /*
         * Step 1. Calculate the gradient of the activation function with
         * respect to the raw output
         */
        for (int i = 0; i < output_size; i++) {
            grad_act[i] = Activation_grad::func_vec[act](out_raw[curr_data][i]);
        }

        /*
         * Step 2. Calculate the gradient of the loss function
         * - If this layer is the last layer, calculate the gradient with
         *   respect to the raw output
         * - Otherwise, use delta from the next layer
         */
        bool is_last_layer = (layer_index == nLayers - 1);
        if (is_last_layer) {
            loss_func->gradient(delta, ground_truth[curr_data],
                                out_act[curr_data], output_size);
            for (int i = 0; i < output_size; i++) {
                delta[i] *= grad_act[i];
            }
        } else {
            transposeMatrix(w_next_transpose, w_next, next_layer_output_size,
                            output_size);
            multiplyMatrix(temp, w_next_transpose, delta_2D[curr_data],
                           output_size, next_layer_output_size,
                           next_layer_output_size, 1);
            for (int i = 0; i < output_size; i++) {
                delta[i] = temp[i][0] * grad_act[i];
            }
        }

        for (int i = 0; i < output_size; i++) {
            delta_2D[curr_data][i][0] = delta[i];
            grad_b[i] += delta[i];
        }

        /*
         * Step 3. Add the gradients of current data sample to the total
         * - Multiply delta with the input2D
         * - Size should be: (output_size, 1) * (1, input_size) = (output_size,
         * input_size)
         */
        for (int i = 0; i < input_size; i++) {
            input2D[0][i] = input[curr_data][i];
        }
        multiplyMatrix(temp_grad_w, delta_2D[curr_data], input2D, output_size,
                       1, 1, input_size);
        clip_grad(temp_grad_w, output_size,
                  input_size);  // needed to prevent exploding gradients
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                grad_w[i][j] += temp_grad_w[i][j];
            }
        }
    }

    /*
     * Step 4. Update the weights and biases
     */
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            const double v =
                momentum * last_v_w[i][j] - lr * grad_w[i][j] / batch_size;
            w[i][j] += v;
            last_v_w[i][j] = v;
        }
        const double v = momentum * last_v_b[i] - lr * grad_b[i] / batch_size;
        b[i] += v;
        last_v_b[i] = v;
    }

    /*
     * delta_2D will be used by the previous layer to calculate its gradients
     */
}

__global__ void cudaComputeActivationGrad(double** grad_act, double** out_raw,
                                          const int act, const int batch_size,
                                          const int output_size) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int curr_data = thread_id / output_size;
    const int i = thread_id % output_size;

    if (curr_data < batch_size && i < output_size) {
        grad_act[curr_data][i] = act_grad_func_vec[act](out_raw[curr_data][i]);
    }
}

void Layer::device_optimize(
    const DeviceLayerOptimizationPointers& device_layer_optimization_pointers,
    const int layer_index, const int nLayers, LossFunction* const loss_func,
    double** const host_ground_truth, double** const device_ground_truth,
    double** const w_next, const int next_layer_output_size,
    double** const input, const double lr, const int batch_size,
    const double momentum) {
    auto [device_grad_act, device_delta, device_w_next_transpose, device_temp,
          device_delta_2D, device_grad_w, device_grad_b, device_temp_grad_w,
          device_input2D] = device_layer_optimization_pointers;

    /*
     * Zero out the gradients.
     */
    CUDA_CHECK_ERROR(
        cudaMemset(device_grad_b, 0, output_size * sizeof(double)));
    double** h_grad_w = new double*[output_size];
    CUDA_CHECK_ERROR(cudaMemcpy(h_grad_w, device_grad_w,
                                output_size * sizeof(double*),
                                cudaMemcpyDeviceToHost));
    for (int i = 0; i < output_size; i++) {
        CUDA_CHECK_ERROR(
            cudaMemset(h_grad_w[i], 0, input_size * sizeof(double)));
    }
    delete[] h_grad_w;

    /*
     * Step 1. Calculate the gradient of the activation function with
     * respect to the raw output
     */
    cudaComputeActivationGrad<<<(batch_size * output_size + 256 - 1) / 256,
                                256>>>(device_grad_act, out_raw, act,
                                       batch_size, output_size);
    cudaDeviceSynchronize();

    /*
     * Step 2. Calculate the gradient of the loss function
     * - If this layer is the last layer, calculate the gradient with
     *   respect to the raw output
     * - Otherwise, use delta from the next layer
     */
    bool is_last_layer = (layer_index == nLayers - 1);
    if (is_last_layer) {
        loss_func->gradient(nullptr, nullptr, nullptr, output_size, use_cuda,
                            batch_size, device_delta, device_ground_truth,
                            out_act);
        device_multiplyArrays<<<(batch_size * output_size + 256 - 1) / 256,
                                256>>>(device_delta, device_grad_act,
                                       batch_size, output_size);
        cudaDeviceSynchronize();
    } else {
        device_transposeMatrix<<<
            (next_layer_output_size * output_size + 256 - 1) / 256, 256>>>(
            device_w_next_transpose, w_next, next_layer_output_size,
            output_size);
        cudaDeviceSynchronize();
        device_multiplyMatrix<<<(batch_size * output_size * 1 + 256 - 1) / 256,
                                256>>>(
            device_temp, device_w_next_transpose, device_delta_2D, output_size,
            next_layer_output_size, next_layer_output_size, 1, batch_size);
        cudaDeviceSynchronize();
        device_multiplyArrays<<<(batch_size * output_size + 256 - 1) / 256,
                                256>>>(device_delta, device_temp,
                                       device_grad_act, batch_size,
                                       output_size);
        cudaDeviceSynchronize();
    }

    device_updateDelta2DAndGradB<<<output_size, 256, 256 * sizeof(double)>>>(
        device_delta_2D, device_delta, device_grad_b, batch_size, output_size);
    cudaDeviceSynchronize();

    /*
     * Step 3. Add the gradients of current data sample to the total
     * - Multiply delta with the input2D
     * - Size should be: (output_size, 1) * (1, input_size) = (output_size,
     * input_size)
     */
    device_setInput2D<<<(batch_size * input_size + 256 - 1) / 256, 256>>>(
        device_input2D, input, batch_size, input_size);
    cudaDeviceSynchronize();
    device_multiplyMatrix<<<
        (batch_size * output_size * input_size + 256 - 1) / 256, 256>>>(
        device_temp_grad_w, device_delta_2D, device_input2D, output_size, 1, 1,
        input_size, batch_size);
    cudaDeviceSynchronize();
    const int clipGradThreadsPerBlock =
        MIN(MULT_32(output_size * input_size), 256);
    const int clipGradSharedMemorySize =
        clipGradThreadsPerBlock * sizeof(double);
    device_clipGrad<<<batch_size, clipGradThreadsPerBlock,
                      clipGradSharedMemorySize>>>(
        device_temp_grad_w, batch_size, output_size, input_size);
    cudaDeviceSynchronize();
    device_updateGradW<<<(output_size * input_size + 256 - 1) / 256, 256>>>(
        device_grad_w, device_temp_grad_w, batch_size, output_size, input_size);
    cudaDeviceSynchronize();

    /*
     * Step 4. Update the weights and biases.
     */
    device_updateWeightsAndBiases<<<(output_size * input_size + 256 - 1) / 256,
                                    256>>>(
        w, last_v_w, device_grad_w, b, last_v_b, device_grad_b, momentum, lr,
        output_size, input_size, batch_size);
    cudaDeviceSynchronize();

    /*
     * delta_2D will be used by the previous layer to calculate its gradients
     */
}

void Layer::setUseCuda(bool new_stat) {
    use_cuda = new_stat;

    // move w, b, out_* into the new place
    if (use_cuda) {
        // allocate memory for w and b
        double** w_cuda = create2DArrayCuda(output_size, input_size, w);

        double* b_cuda = nullptr;
        cudaMallocAsync(&b_cuda, output_size * sizeof(double), 0);
        cudaMemcpyAsync(b_cuda, b, output_size * sizeof(double),
                        cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // move out_raw and out_act to the GPU memory
        double** out_raw_cuda =
            create2DArrayCuda(batch_size, output_size, out_raw);
        double** out_act_cuda = out_raw_cuda;
        if (act != ACT_NONE) {
            double** out_act_cuda =
                create2DArrayCuda(batch_size, output_size, out_act);
        }

        // free w, b, out_raw, out_act
        free2DArray(w, output_size);
        delete[] b;
        free2DArray(out_raw, batch_size);
        if (act != ACT_NONE) {
            free2DArray(out_act, batch_size);
        }

        // assign new values
        w = w_cuda;
        b = b_cuda;
        out_raw = out_raw_cuda;
        out_act = out_act_cuda;
    } else {
        // move w, b, out_raw, out_act back to the CPU memory
        // allocate memory for w and b
        double** w_cpu = new double*[output_size];
        for (int i = 0; i < output_size; i++) {
            w_cpu[i] = new double[input_size];
        }

        double* b_cpu = new double[output_size];

        double** out_raw_cpu = new double*[batch_size];
        for (int i = 0; i < batch_size; i++) {
            out_raw_cpu[i] = new double[output_size];
        }

        double** out_act_cpu = out_raw_cpu;
        if (act != ACT_NONE) {
            out_act_cpu = new double*[batch_size];
            for (int i = 0; i < batch_size; i++) {
                out_act_cpu[i] = new double[output_size];
            }
        }

        // copy the data from the GPU memory to the CPU memory, and free gpu
        // memory
        free2DArrayCuda(w, output_size, w_cpu, input_size);
        cudaMemcpy(b_cpu, b, output_size * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaFree(b);
        free2DArrayCuda(out_raw, batch_size, out_raw_cpu, output_size);
        if (act != ACT_NONE) {
            free2DArrayCuda(out_act, batch_size, out_act_cpu, output_size);
        }

        // assign new values
        w = w_cpu;
        b = b_cpu;
        out_raw = out_raw_cpu;
        out_act = out_act_cpu;
    }
}

/**
 * @brief Save the layer to a file.
 *
 * Saves the weights and other parameters of the layer to a file in binary
 * format.
 *
 * After saving, the file will contain the following data in order of
 * appearance:
 * - input_size (unsigned long long)
 * - output_size (unsigned long long)
 * - act (int)
 * - w (output_size * input_size)
 * - b (output_size)
 */
void Layer::saveLayer(FILE* file) {
    fwrite(&input_size, sizeof(unsigned long long), 1, file);
    fwrite(&output_size, sizeof(unsigned long long), 1, file);
    fwrite(&act, sizeof(int), 1, file);

    if (use_cuda) {
        double** host_w = create2DArray(output_size, input_size);
        copy2DArrayFromDevice(host_w, w, output_size, input_size);
        double* host_b = new double[output_size];
        cudaMemcpy(host_b, b, output_size * sizeof(double),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < output_size; i++) {
            fwrite(host_w[i], sizeof(double), input_size, file);
        }
        fwrite(host_b, sizeof(double), output_size, file);

        free2DArray(host_w, output_size);
        delete[] host_b;
    } else {
        for (int i = 0; i < output_size; i++) {
            fwrite(w[i], sizeof(double), input_size, file);
        }
        fwrite(b, sizeof(double), output_size, file);
    }
}

void load_weight(double** w, double* b, int input_size, int output_size,
                 FILE* file) {
    for (int i = 0; i < output_size; i++) {
        if (fread(w[i], sizeof(double), input_size, file) != input_size) {
            std::cerr << "Error reading weights from file" << std::endl;
            exit(1);
        }
    }

    if (fread(b, sizeof(double), output_size, file) != output_size) {
        std::cerr << "Error reading biases from file" << std::endl;
        exit(1);
    }
}

void Layer::loadLayer(FILE* file) {
    /*
     * Precondition: w and b are already in memory.
     * Dimensions of w: output_size x input_size
     * Dimensions of b: output_size
     */

    if (use_cuda) {
        double** host_w = create2DArray(output_size, input_size);
        double* host_b = new double[output_size];

        load_weight(host_w, host_b, input_size, output_size, file);

        if (w != nullptr) {
            free2DArrayCuda(w, output_size);
        }
        if (b != nullptr) {
            cudaFree(b);
        }

        w = create2DArrayCuda(output_size, input_size, host_w);
        cudaMalloc(&b, output_size * sizeof(double));
        cudaMemcpy(b, host_b, output_size * sizeof(double),
                   cudaMemcpyHostToDevice);

        free2DArray(host_w, output_size);
        delete[] host_b;
    } else {
        load_weight(w, b, input_size, output_size, file);
    }
}