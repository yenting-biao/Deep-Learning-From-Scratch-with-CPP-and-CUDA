#ifndef LAYER_HPP
#define LAYER_HPP

#include <cstdio>

#include "loss.hpp"
#include "optimize.hpp"
#define ACT_NONE 0
#define ACT_RELU 1
#define ACT_SIGMOID 2

class Layer {
   public:
    Layer(unsigned long long input_size = 0, unsigned long long output_size = 0,
          int act = ACT_NONE, int batch_size = 1, bool use_cuda = false,
          double** weights = nullptr,
          double* biases =
              nullptr);  // input_size, output_size, activation function
    ~Layer();

    void forward(double** input, int batch_size, double** output = nullptr);
    void host_optimize(
        const HostLayerOptimizationPointers& layer_optimization_pointers,
        int layer_index, int nLayers, LossFunction* loss_func,
        double** ground_truth, double** w_next, int next_layer_output_size,
        double** input, double lr, int batch_size, double momentum = 0.0);
    void device_optimize(const DeviceLayerOptimizationPointers&
                             device_layer_optimization_pointers,
                         int layer_index, int nLayers, LossFunction* loss_func,
                         double** host_ground_truth,
                         double** device_ground_truth, double** w_next,
                         int next_layer_output_size, double** input, double lr,
                         int batch_size, double momentum = 0.0);

    void setUseCuda(bool new_stat);

    void saveLayer(FILE* file);
    void loadLayer(FILE* file);

    unsigned long long input_size;
    unsigned long long output_size;
    const int batch_size;

    double** out_raw;  // raw output after Linear layer
    double** out_act;  // activated output after activation function

    double** w;
    double* b;

    double** last_v_w;
    double* last_v_b;

   private:
    int act;  // activation function
    bool use_cuda = false;
};

#endif  // LAYER_HPP