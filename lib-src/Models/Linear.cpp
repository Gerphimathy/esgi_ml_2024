#include <iostream>
#include "Linear.hpp"
#include <vector>

namespace MachineLearning {
    Linear::Linear(int n, double b = 0) {
        N = n;
        bias = b;
        weights = std::vector<double>(N, 0);
    }

    void Linear::train(const std::vector<std::vector<double>> &X, const std::vector<double> &Y,
                       double learning_rate, int epochs) {
        //Initialize weights
        for (int i = 0; i < N; i++) {
            weights[i] = 0;
        }
        bias = 0;

        //Train
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < X.size(); j++) {
                double p = predict(X[j]);
                float err = Y[j] - p;
                for (int k = 0; k < N; ++k) {
                    weights[k] += err * X[j][k] * learning_rate;
                }
                bias += err * learning_rate;
            }
        }
    }

    double Linear::predict(const std::vector<double> &x)const{
        double pred = 0;
        for (int i = 0; i < N; i++) {
            pred += weights[i] * x[i];
        }
        pred += bias;
        return pred;
    }

    bool Linear::setWeight(double w, int i) {
        if(i < 0 || i >= N) return false;
        weights[i] = w;
        return true;
    }

    void Linear::setBias(double b) {
        bias = b;
    }
}