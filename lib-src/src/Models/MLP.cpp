#include <cstdlib>
#include <valarray>
#include <iostream>
#include "MLP.hpp"
#include "ranges.hpp"

namespace MachineLearning {
    MLP::MLP(const std::vector<int> &npl, Activation a) {
        if(npl.empty()) return;

        activation = a;

        dimensions = npl;
        L = npl.size()-1;

        weights = {};
        X = {};
        deltas = {};

        for(int l = 0; l < npl.size(); l++){
            X.emplace_back();
            deltas.emplace_back();
            for(int j = 0; j < dimensions[l]+1; j++){
                deltas[l].emplace_back(0.0);
                if(j == 0) X[l].emplace_back(1.0);
                else X[l].emplace_back(0.0);
            }

            weights.emplace_back();
            if(l == 0) continue;
            for(int i = 0; i < dimensions[l-1]+1; i++){
                weights[l].emplace_back();
                for(int j = 0; j < dimensions[l]+1; j++){
                    if(j == 0) weights[l][i].push_back(0.0);
                    else weights[l][i].push_back(randomFloat(-1.0, 1.0));
                }
            }
        }
    }

    double MLP::activate(double x) {
        switch(activation){
            case SIGMOID:
                if(x < -45.0) return 0.0;
                else if(x > 45.0) return 1.0;
                else return 1.0 / (1.0 + exp(-x));
            case TANH:
                return tanh(x);
            case RELU:
                return x > 0 ? x : 0;
            case LEAKY_RELU:
                return x > 0 ? x : 0.01 * x;
            case SOFTMAX:
                if(x > 700.0) x = 700;
                return exp(x);
            default:
                return 0.0;
        }
    }

    void MLP::propagate_forward(const std::vector<double> &inputs, bool classify) {
        for(int j = 1; j < dimensions[0]+1; j++){
            X[0][j] = inputs[j-1];
        }

        for(int l = 1; l < dimensions.size(); l++){
            for(int j = 1; j < dimensions[l]+1; j++){
                double sum = 0.0;
                for(int i = 0; i < dimensions[l-1]+1; i++){
                    sum += weights[l][i][j] * X[l-1][i];
                }
                if(l < L or classify) sum = activate(sum);

                X[l][j] = sum;
            }
        }
    }

    std::vector<double> MLP::predict(const std::vector<double> &inputs, bool classify) {
        propagate_forward(inputs, classify);
        std::vector<double> outputs;
        for(int j = 1; j < dimensions[L]+1; j++){
            outputs.push_back(X[L][j]);
        }
        return outputs;
    }

    void MLP::process_weights(const double &training_rate){
        for (int l = 1; l < L+1; l++){
            for(int i = 0; i < dimensions[l-1]+1; i++){
                for(int j = 1; j < dimensions[l]+1; j++){
                    weights[l][i][j] -= training_rate * X[l-1][i] * deltas[l][j];
                }
            }
        }
    }

    void MLP::propagate_backwards(const std::vector<double> &y, bool classify){
        for(int j = 1; j < dimensions[L]+1; j++){
            deltas[L][j] = X[L][j] - y[j-1];
            if(classify) deltas[L][j] *= (1 - X[L][j]*X[L][j]);
        }

        for(int l = L; l > 1; l--){
            for(int i = 0; i < dimensions[l-1]+1; i++){
                double sum = 0.0;
                for(int j = 1; j < dimensions[l]+1; j++){
                    sum += weights[l][i][j] * deltas[l][j];
                }
                deltas[l-1][i] = (1 - X[l-1][i]*X[l-1][i]) * sum;
            }
        }
    }

    bool MLP::train(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, bool classify,
                    double training_rate, unsigned int epochs, bool verbose,
                    Sampling sampling, unsigned int batch_size
                    ) {
        if(X.size() != Y.size()) return false;
        if(sampling == MINI_BATCH_GRADIANT_DESCENT){
            if(batch_size == 0) return false;
            if(batch_size > X.size()) return false;
        }
        for(int e = 0; e < epochs; e++){
            if(verbose) std::cout << "Epoch " << e+1 << "/" << epochs << "\n";

            std::vector<double> x;
            std::vector<double> y;
            int k;
            switch (sampling) {
                case RANDOM:
                    k = randomInt(0, X.size()-1);
                    x = X[k];
                    y = Y[k];
                    propagate_forward(x, classify);
                    propagate_backwards(y, classify);
                    process_weights(training_rate);
                    break;
                case BATCH_GRADIANT_DESCENT:
                    for (int i = 0; i < X.size(); ++i) {
                        x = X[i];
                        y = Y[i];
                        propagate_forward(x, classify);
                        propagate_backwards(y, classify);
                    }
                    process_weights(training_rate);
                    break;
                case STOCHASTIC_GRADIANT_DESCENT:
                    for (int i = 0; i < X.size(); i++) {
                        x = X[i];
                        y = Y[i];
                        propagate_forward(x, classify);
                        propagate_backwards(y, classify);
                        process_weights(training_rate);
                    }
                    break;
                case MINI_BATCH_GRADIANT_DESCENT:
                    k = randomInt(0, X.size()-batch_size);
                    for (int i = k; i < k + batch_size; ++i) {
                        x = X[i];
                        y = Y[i];
                        propagate_forward(x, classify);
                        propagate_backwards(y, classify);
                    }
                    break;
            }
        }
        return true;
    }

    std::vector<std::vector<std::vector<double>>> MLP::get_weights() const {
        return weights;
    }

    std::vector<int> MLP::get_dimensions() const {
        return dimensions;
    }
}