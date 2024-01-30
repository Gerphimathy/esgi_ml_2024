///Main.cpp is for testing purposes (finding memleaks) only, the C++ code is to be made into a dll

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include "Models/MLP.hpp"
#include "ranges.hpp"

int main() {
    int n_points = 500;
    int epochs = 100;
    double training_rate = 0.00001;
    MachineLearning::Activation activation = MachineLearning::TANH;
    MachineLearning::Sampling sampling = MachineLearning::STOCHASTIC_GRADIANT_DESCENT;
    bool verbose = false;
    bool classify = true;
    std::vector<int> layers = {2, 5, 3};

    srand(time(nullptr));

    MachineLearning::MLP mlp(layers, activation);

    auto X = std::vector<std::vector<double>>(n_points);
    auto Y = std::vector<std::vector<double>>(n_points);

    for (int i = 0; i < n_points; ++i) {
        X[i] = std::vector<double>(2);
        X[i][0] = MachineLearning::randomFloat(-1.0, 1.0);
        X[i][1] = MachineLearning::randomFloat(-1.0, 1.0);
    }

    for (int i = 0; i < n_points; ++i) {
        auto p = X[i];
        if(-p[0] - p[1] - 0.5 > 0 && p[1] < 0 && p[0] - p[1] - 0.5 < 0) Y[i] = {1.0, 0.0, 0.0};
        else if(-p[0] - p[1] - 0.5 < 0 && p[1] > 0 && p[0] - p[1] - 0.5 < 0) Y[i] = {0.0, 1.0, 0.0};
        else if(-p[0] - p[1] - 0.5 < 0 && p[1] < 0 && p[0] - p[1] - 0.5 > 0) Y[i] = {0.0, 0.0, 1.0};
        else Y[i] = {0.0, 0.0, 0.0};
    }

    mlp.train(X, Y, classify, training_rate, epochs, verbose, sampling);

    mlp.serialize("test.txt");

    auto mlp2 = MachineLearning::MLP("test.txt");

    mlp2.serialize("test2.txt");

    return 0;
}