#ifndef MODEL_HPP
#define MODEL_HPP

#include "layer.hpp"
#include "loss.hpp"
#include "optimize.hpp"
#include "utils.hpp"

#define DEFAULT_SEED 42

class Model {
   public:
    Model(double lr, double momentum, LossFunction* loss_func,
          unsigned int seed = DEFAULT_SEED, int batch_size = 1);
    ~Model();

    void forward(double** input, int batch);
    void optimize(double** ground_truth, double** input);
    void addLayer(int input_size, int output_size, int activation = ACT_NONE);

    void setUseCuda(bool new_stat);

    double calculateLoss(double** const gt, const int batch_size, const int n);

    void fecthOutput(double** output, int batch_size);

    void saveModel(const char* filename);
    void loadModel(const char* filename);

   private:
    const double lr;
    const double momentum;
    const int batch_size;
    int curr_batch_size;
    LossFunction* const loss_func;

    Layer** layers = nullptr;  // [i] => ptr to the i-th layer
    int nLayers = 0;
    int capacity = 0;

    // about optimization
    double* loss_grad;  // gradient of the loss function with respect to the
                        // output of the last layer

    // used when optimizing
    // user should not free this memory
    double** input;
    double** ground_truth;

    // if use_cuda is true, the model will use CUDA to optimize
    // otherwise, the model will use CPU by default
    bool use_cuda = false;
    int last_batch_size;

    // for Layer::optimize()
    int max_input_size = 0;
    int max_output_size = 0;
    HostLayerOptimizationPointers host_layer_optimization_pointers;
    DeviceLayerOptimizationPointers device_layer_optimization_pointers;
    double** device_ground_truth = nullptr;
};

#endif  // MODEL_HPP
