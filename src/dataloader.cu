#include "dataloader.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

DataLoader::DataLoader(double** const input, double** const output,
                       const int num_rows, const int batch_size)
    : num_batches((num_rows + batch_size - 1) / batch_size),
      input(input),
      output(output),
      num_rows(num_rows),
      batch_size(batch_size) {
    batch_input = new double*[batch_size];
    batch_output = new double*[batch_size];
}

DataLoader::~DataLoader() {
    delete[] batch_input;
    delete[] batch_output;
}

DataLoader::Iterator::Iterator(const DataLoader& dataloader,
                               const int curr_batch_index)
    : dataloader(dataloader), curr_batch_index(curr_batch_index) {}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return curr_batch_index != other.curr_batch_index;
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    curr_batch_index++;
    return *this;
}

DataLoader::Iterator::Batch DataLoader::Iterator::operator*() const {
    const int current_batch_size =
        MIN(dataloader.batch_size,
            dataloader.num_rows - curr_batch_index * dataloader.batch_size);
    for (int i = 0; i < current_batch_size; i++) {
        dataloader.batch_input[i] =
            dataloader.input[curr_batch_index * dataloader.batch_size + i];
        dataloader.batch_output[i] =
            dataloader.output[curr_batch_index * dataloader.batch_size + i];
    }
    return Batch{dataloader.batch_input, dataloader.batch_output,
                 curr_batch_index};
}

DataLoader::Iterator DataLoader::begin() { return Iterator(*this, 0); }

DataLoader::Iterator DataLoader::end() { return Iterator(*this, num_batches); }