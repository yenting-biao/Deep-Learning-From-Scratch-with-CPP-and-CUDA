#ifndef LOSS_HPP
#define LOSS_HPP

class LossFunction {
   public:
    virtual double compute(const double* ground_truth, const double* prediction,
                           int n) = 0;
    virtual double compute_OnDev(double** device_ground_truth,
                                 double** device_prediction, const int n,
                                 const int batch_size) = 0;
    virtual void gradient(double* grad, const double* ground_truth,
                          const double* prediction, int n,
                          bool use_cuda = false, int batch_size = 1,
                          double** device_delta = nullptr,
                          double** device_ground_truth = nullptr,
                          double** out_act = nullptr) = 0;
};

class MSELoss : public LossFunction {
   public:
    double compute(const double* ground_truth, const double* prediction,
                   int n) override;
    double compute_OnDev(double** device_ground_truth,
                         double** device_prediction, const int n,
                         const int batch_size) override;
    void gradient(double* grad, const double* ground_truth,
                  const double* prediction, int n, bool use_cuda = false,
                  int batch_size = 1, double** device_delta = nullptr,
                  double** device_ground_truth = nullptr,
                  double** out_act = nullptr) override;
};

class CrossEntropyLoss : public LossFunction {
   public:
    double compute(const double* ground_truth, const double* prediction,
                   int n) override;
    double compute_OnDev(double** device_ground_truth,
                         double** device_prediction, const int n,
                         const int batch_size) override;
    void gradient(double* grad, const double* ground_truth,
                  const double* prediction, int n, bool use_cuda = false,
                  int batch_size = 1, double** device_delta = nullptr,
                  double** device_ground_truth = nullptr,
                  double** out_act = nullptr) override;
};

#endif  // LOSS_HPP
