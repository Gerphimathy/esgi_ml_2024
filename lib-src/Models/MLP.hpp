#pragma once

#include <vector>

namespace MachineLearning {
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
            void propagate_backwards(const std::vector<double> &y, bool classify);
            void process_weights(const double &training_rate);

        public:
            MLP(const std::vector<int>& layers, Activation a);
            std::vector<double> predict(const std::vector<double>& inputs, bool classify);
            bool train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y,
                       bool classify, double training_rate, unsigned int epochs, bool verbose = false,
                       Sampling sampling = RANDOM, unsigned int batch_size = 0);
    };
}