#include <stdlib.h>

#include <algorithm>
#include <iostream>

#include "model.hpp"
#include "optimize.hpp"
#include "utils.hpp"

Model::Model(const double lr, const double momentum,
             LossFunction* const loss_func, unsigned int seed, int batch_size)
    : lr(lr), momentum(momentum), batch_size(batch_size), loss_func(loss_func) {
    if (seed == DEFAULT_SEED)
        printf("No seed provided, using default seed %d.\n", DEFAULT_SEED);
    srand(seed);
    last_batch_size = -1;
    input = nullptr;
    if (use_cuda) {
        device_ground_truth =
            create2DArrayCuda(batch_size, max_output_size, nullptr);
    }
}

Model::~Model() {
    for (int i = 0; i < nLayers; i++) {
        delete layers[i];
    }

    delete[] layers;
    if (use_cuda && input != nullptr) {
        free2DArrayCuda(input, last_batch_size);
    }

    if (host_layer_optimization_pointers.grad_act != nullptr) {
        std::cerr << "Model.~Model(): freeing host memory" << std::endl;
        _host_freeMemoryForLayerOptimizationPointers(
            host_layer_optimization_pointers, max_output_size, batch_size);
    }
    if (device_layer_optimization_pointers.grad_act != nullptr) {
        std::cerr << "Model.~Model(): freeing device memory" << std::endl;
        _device_freeMemoryForLayerOptimizationPointers(
            device_layer_optimization_pointers, max_output_size, batch_size);
    }
    if (device_ground_truth != nullptr) {
        free2DArrayCuda(device_ground_truth, batch_size);
    }
}

void Model::forward(double** const input_data, const int batch_size) {
    curr_batch_size = batch_size;
    if (use_cuda) {
        if (batch_size != last_batch_size) {  // reallocate memory for input
            if (input) free2DArrayCuda(input, last_batch_size);

            input = create2DArrayCuda(batch_size, layers[0]->input_size,
                                      input_data);

            last_batch_size = batch_size;
        } else {
            copy2DArrayToDevice(input, input_data, batch_size,
                                layers[0]->input_size);
        }
    } else {
        input = input_data;
    }

    layers[0]->forward(input, batch_size);

    for (int i = 1; i < nLayers; i++) {
        layers[i]->forward(layers[i - 1]->out_act, batch_size);
    }
}

void Model::setUseCuda(bool new_stat) {
    if (use_cuda != new_stat) {
        /*
         * Free existing memory. The new memory will be allocated in
         * Model.optimize() instead of here to avoid repeatedly freeing and
         * allocating memory if the user changes use_cuda multiple times.
         */
        if (!use_cuda && host_layer_optimization_pointers.grad_act != nullptr) {
            std::cerr << "Model.setUseCuda(): freeing host memory" << std::endl;
            _host_freeMemoryForLayerOptimizationPointers(
                host_layer_optimization_pointers, max_output_size, batch_size);
        } else if (use_cuda &&
                   device_layer_optimization_pointers.grad_act != nullptr) {
            std::cerr << "Model.setUseCuda(): freeing device memory"
                      << std::endl;
            _device_freeMemoryForLayerOptimizationPointers(
                device_layer_optimization_pointers, max_output_size,
                batch_size);
        }

        use_cuda = new_stat;

        for (int i = 0; i < nLayers; i++) {
            layers[i]->setUseCuda(use_cuda);
        }
    }
}

double Model::calculateLoss(double** const gt, const int batch_size,
                            const int n) {
    double total_loss = 0.0;

    double** pred = layers[nLayers - 1]->out_act;

    if (!use_cuda) {
        ground_truth = gt;

        for (int i = 0; i < batch_size; i++) {
            total_loss += loss_func->compute(gt[i], pred[i], n);
        }
    } else {
        if (device_ground_truth == nullptr) {
            device_ground_truth =
                create2DArrayCuda(batch_size, max_output_size, gt);
        } else {
            copy2DArrayToDevice(device_ground_truth, gt, batch_size,
                                max_output_size);
        }

        total_loss =
            loss_func->compute_OnDev(device_ground_truth, pred, n, batch_size);
    }

    return total_loss / batch_size;
}

void Model::optimize(double** ground_truth, double** input) {
    if (!use_cuda && host_layer_optimization_pointers.grad_act == nullptr) {
        std::cerr << "Model.optimize(): allocating host memory" << std::endl;
        _host_allocateMemoryForLayerOptimizationPointers(
            host_layer_optimization_pointers, max_output_size, max_input_size,
            batch_size);
    } else if (use_cuda &&
               device_layer_optimization_pointers.grad_act == nullptr) {
        std::cerr << "Model.optimize(): allocating device memory" << std::endl;
        _device_allocateMemoryForLayerOptimizationPointers(
            device_layer_optimization_pointers, max_output_size, max_input_size,
            batch_size);
    }

    for (int i = nLayers - 1; i >= 0; i--) {
        if (!use_cuda) {
            layers[i]->host_optimize(
                host_layer_optimization_pointers, i, nLayers, loss_func,
                ground_truth,
                (i == nLayers - 1 ? nullptr
                                  : static_cast<double**>(layers[i + 1]->w)),
                (i == nLayers - 1 ? 0 : layers[i + 1]->output_size),
                (i == 0 ? static_cast<double**>(input)
                        : layers[i - 1]->out_act),
                lr, batch_size, momentum);
        } else {
            layers[i]->device_optimize(
                device_layer_optimization_pointers, i, nLayers, loss_func,
                ground_truth, device_ground_truth,
                (i == nLayers - 1 ? nullptr
                                  : static_cast<double**>(layers[i + 1]->w)),
                (i == nLayers - 1 ? 0 : layers[i + 1]->output_size),
                (i == 0 ? this->input : layers[i - 1]->out_act), lr, batch_size,
                momentum);
        }
    }
}

void expandArray(Layer**& arr, int& capacity) {
    // expand the array under the same rule as std::vector
    // Note: only called when (n == capacity)

    if (capacity == 0) {
        // allocate memory for the array
        arr = new Layer*[1];
        capacity = 1;
    } else {
        // expand the array
        Layer** newArr = new Layer*[capacity * 2];

        for (int i = 0; i < capacity; i++) {
            newArr[i] = arr[i];
        }

        delete[] arr;

        arr = newArr;

        capacity *= 2;
    }
}

void Model::addLayer(const int input_size, const int output_size,
                     const int activation) {
    // add a new layer to the model

    // check the dimension of the input_size
    if (nLayers > 0 && input_size != layers[nLayers - 1]->output_size) {
        std::cerr << "<ERROR> Model::addLayer(): The input size (" << input_size
                  << ") of the new layer is not compatible with the output "
                     "size of the last layer (layer "
                  << nLayers - 1
                  << ", output_size =  " << layers[nLayers - 1]->output_size
                  << ").\n";
        exit(1);
    }

    if (nLayers == capacity) {
        expandArray(layers, capacity);
    }

    layers[nLayers] =
        new Layer(input_size, output_size, activation, batch_size, use_cuda);

    nLayers++;

    // for Layer::optimize()
    max_input_size = std::max(max_input_size, input_size);
    max_output_size = std::max(max_output_size, output_size);
}

void Model::saveModel(const char* filename) {
    FILE* file = fopen(filename, "wb");

    if (file == nullptr) {
        std::cerr << "<ERROR> Model::saveModel(): Cannot open file " << filename
                  << ".\n";
        exit(1);
    }

    // write the number of layers
    fwrite(&nLayers, sizeof(int), 1, file);

    // write the layers
    for (int i = 0; i < nLayers; i++) {
        // write the input_size, output_size, activation function
        layers[i]->saveLayer(file);
    }

    fclose(file);
    std::cout << "<Info> Model saved to " << filename << std::endl;
}

void Model::loadModel(const char* filename) {
    FILE* file = fopen(filename, "rb");

    if (file == nullptr) {
        std::cerr << "<ERROR> Model::loadModel(): Cannot open file " << filename
                  << ".\n";
        exit(1);
    }

    // Clear existing layers
    if (layers != nullptr) {
        std::cerr
            << "<WARNING> Model::loadModel(): Clearing existing layers.\n";
        for (int i = 0; i < nLayers; i++) {
            delete layers[i];
        }
        delete[] layers;
        layers = nullptr;
        nLayers = 0;
        capacity = 0;
    }

    // read the number of layers
    int numLayers;
    if (fread(&numLayers, sizeof(int), 1, file) != 1) {
        std::cerr << "Error reading in Model::loadModel()" << std::endl;
        exit(1);
    }

    // read the layers
    for (int i = 0; i < numLayers; i++) {
        unsigned long long input_size, output_size;
        int act;
        if (fread(&input_size, sizeof(unsigned long long), 1, file) != 1) {
            std::cerr << "Error reading in Model::loadModel()" << std::endl;
            exit(1);
        }
        if (fread(&output_size, sizeof(unsigned long long), 1, file) != 1) {
            std::cerr << "Error reading in Model::loadModel()" << std::endl;
            exit(1);
        }
        if (fread(&act, sizeof(int), 1, file) != 1) {
            std::cerr << "Error reading in Model::loadModel()" << std::endl;
            exit(1);
        }
        this->addLayer(input_size, output_size, act);
        layers[i]->loadLayer(file);
    }

    fclose(file);

    std::cout << "<INFO> Model loaded from " << filename << std::endl;
}

void Model::fecthOutput(double** output, int batch_size) {
    if (output != nullptr) {
        if (use_cuda) {
            copy2DArrayFromDevice(output, layers[nLayers - 1]->out_act,
                                  batch_size, layers[nLayers - 1]->output_size);
        } else {
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < layers[nLayers - 1]->output_size; j++) {
                    output[i][j] = layers[nLayers - 1]->out_act[i][j];
                }
            }
        }
    }
}