#pragma once

#include <vector>

namespace MachineLearning {
    std::vector<double> s_normalize(const std::vector<double> &x, std::vector<std::pair<double, double>> minmax);
    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>> &X, std::vector<std::pair<double, double>> minmax);
    double rescale(double x, double old_min, double old_max, double new_min, double new_max);

    enum Activation {
        SIGMOID,
        TANH,
        RELU,
        LEAKY_RELU,
        SOFTMAX
    };

    enum Sampling{
        RANDOM,
        BATCH_GRADIANT_DESCENT,
        STOCHASTIC_GRADIANT_DESCENT,
        MINI_BATCH_GRADIANT_DESCENT
    };

    class MLP {
        private:
            int L;
            std::vector<std::vector<std::vector<double>>> weights;
            std::vector<int> dimensions;
            std::vector<std::vector<double>> X;
            std::vector<std::vector<double>> deltas;
            Activation activation = SIGMOID;

            void propagate_forward(const std::vector<double>& inputs, bool classify);
            double activate(double x);
            void propagate_backwards(std::vector<double> &y, bool classify);
            void process_weights(double training_rate);

        public:
            MLP(const std::vector<int>& layers, Activation a);
            std::vector<double> predict(const std::vector<double>& inputs, bool classify);
            bool train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y,
                       bool classify, double training_rate, unsigned int epochs, bool verbose = false,
                       Sampling sampling = RANDOM, unsigned int batch_size = 0);
    };
}