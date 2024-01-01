#pragma once

#include <vector>

namespace MachineLearning {
    std::vector<double> s_normalize(const std::vector<double> &x, std::vector<std::pair<double, double>> minmax);
    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>> &X, std::vector<std::pair<double, double>> minmax);

    class MLP {
        private:
            int L;
            std::vector<std::vector<std::vector<double>>> weights;
            std::vector<int> dimensions;
            std::vector<std::vector<double>> X;
            std::vector<std::vector<double>> deltas;

            void propagate(const std::vector<double>& inputs, bool classify);

        public:
            MLP(const std::vector<int>& layers);
            std::vector<double> predict(const std::vector<double>& inputs, bool classify);
            void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y,
                       bool classify, double training_rate, int epochs, bool verbose = false);
    };
}