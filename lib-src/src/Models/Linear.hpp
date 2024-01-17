#pragma once

#include <vector>

namespace MachineLearning {
    class Linear {
        private:
            std::vector<double> weights;
            double bias;
            int N;

        public:
            Linear(int n, double b);
            void train(const std::vector<std::vector<double>>& X, const std::vector<double>& Y,
                       double learning_rate, int epochs);
            double predict(const std::vector<double>& x)const;
            bool setWeight(double w, int i);
            void setBias(double b);
            std::vector<double> getWeights()const{return weights;}
    };
}