#ifndef DATALOADER_HPP
#define DATALOADER_HPP

class DataLoader {
   public:
    DataLoader(double** input, double** output, int num_rows, int batch_size);
    ~DataLoader();
    class Iterator;
    Iterator begin();
    Iterator end();
    const int num_batches;

   private:
    double** const input;
    double** const output;
    double** batch_input;
    double** batch_output;
    const int num_rows;
    const int batch_size;
    friend class Iterator;
};

class DataLoader::Iterator {
   public:
    Iterator(const DataLoader& dataloader, int curr_batch_index);

    bool operator!=(const Iterator& other) const;

    Iterator& operator++();

    struct Batch {
        double** const input;
        double** const output;
        const int curr_batch_index;
    };
    Batch operator*() const;

   private:
    const DataLoader& dataloader;
    int curr_batch_index;
};

#endif  // DATALOADER_HPP