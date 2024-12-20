#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

#include "src/dataloader.hpp"
#include "src/loss.hpp"
#include "src/model.hpp"
#include "src/utils.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void load_data(const char* const filename, double**& input, double**& output,
               int& num_rows, int& input_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "[ERROR]: Failed to open file\n");
        exit(1);
    }

    char line[1024];
    int rows = 0;
    int cols = 0;
    double** data = nullptr;

    bool first_line = false;

    while (fgets(line, sizeof(line), file)) {
        if (!first_line) {
            first_line = true;
            continue;
        }

        char* token = strtok(line, ",");
        int col_count = 0;
        double* row_data =
            (double*)malloc(sizeof(double) * 100);  // assuming max 100 columns

        while (token) {
            row_data[col_count++] = atof(token);
            token = strtok(NULL, ",");
        }

        if (cols == 0) {
            cols = col_count;
        }

        data = (double**)realloc(data, sizeof(double*) * (rows + 1));
        data[rows++] = row_data;
    }

    fclose(file);

    // takes the second column as the output, and removes the first column
    // other columns are used as input (data[i][2:])
    input = new double*[rows];
    input_size = cols - 2;
    num_rows = rows;
    for (int i = 0; i < rows; i++) {
        input[i] = new double[cols - 2];
        for (int j = 2; j < cols; j++) {
            input[i][j - 2] = data[i][j];
        }
    }

    output = new double*[rows];
    for (int i = 0; i < rows; i++) {
        output[i] = new double[1];
        output[i][0] = data[i][1];
    }

    free2DArray(data, rows);
}

int main() {
    /*
     * USAGE:
     * 1. Put input and output features in two separate 2D arrays
     * 2. Define hyperparameters
     * 3. Create a model object with hyperparameters
     * 4. If using GPU, set use_cuda BEFORE adding layers
     * 5. Add layers to the model
     * 6. Set up a DataLoader object with desired batch size
     * 7. Train the model
     *
     * Example usage follows below.
     */

    /*
     * Step 1: Load data into two 2D arrays.
     */
    double **input, **output;
    int num_rows, input_size;
    load_data("./data/processed_data/weatherHistory.csv", input, output,
              num_rows, input_size);

    /*
     * Step 2: Define hyperparameters.
     */
    const int num_epochs = 20;
    const int batch_size = 512;  // Larger batch sizes favor GPU
    const double lr = 0.001;
    const double momentum = 0;
    const unsigned int seed = 0;
    const int hidden_size = 128;
    const int output_size = 1;
    const char* filename = "regression_model.bin";
    class MSELoss loss_func;

    /*
     * Step 3: Create a model object with hyperparameters.
     */
    Model model(lr, momentum, &loss_func, seed, batch_size);

    /*
     * Step 4: If using GPU, set use_cuda BEFORE loading model or adding layers.
     * By default, use_cuda is false.
     */
    model.setUseCuda(true);

    /*
     * Optional: Load a model from a file.
     * Please load the model AFTER setting use_cuda.
     */
    // model.loadModel(filename);

    /*
     * Step 5: Add layers to the model.
     */
    model.addLayer(input_size, hidden_size, ACT_RELU);
    model.addLayer(hidden_size, hidden_size, ACT_RELU);
    model.addLayer(hidden_size, hidden_size, ACT_RELU);
    model.addLayer(hidden_size, output_size, ACT_NONE);

    /*
     * Step 6: Set up a DataLoader object with desired batch size.
     */
    DataLoader dataloader(input, output, num_rows, batch_size);

    /*
     * Step 7: Train the model.
     */
    for (int current_epoch = 1; current_epoch <= num_epochs; current_epoch++) {
        auto epoch_start_time = std::chrono::high_resolution_clock::now();
        double total_loss = 0.0;

        for (auto batch : dataloader) {
            model.forward(batch.input, batch_size);
            const double current_loss =
                model.calculateLoss(batch.output, batch_size, output_size);
            total_loss += current_loss;

            model.optimize(batch.output, batch.input);

            print_progress_bar(
                current_epoch, num_epochs, batch.curr_batch_index + 1,
                dataloader.num_batches, current_loss, epoch_start_time);
        }
        printf(" | Mean Loss: %f\n", total_loss / dataloader.num_batches);
    }

    /*
     * Optional: Save the model to a file.
     */
    model.saveModel(filename);

    free2DArray(input, num_rows);
    free2DArray(output, num_rows);
    return 0;
}
