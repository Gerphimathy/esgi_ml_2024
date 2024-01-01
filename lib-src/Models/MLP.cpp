#include <cstdlib>
#include <valarray>
#include <iostream>
#include "MLP.hpp"

namespace MachineLearning {
    float randomFloat(float a, float b) {
        return ((b - a) * ((float)rand() / RAND_MAX)) + a;
    }
    int randomInt(int a, int b) {
        return (int) ((b - a) * ((float)rand() / RAND_MAX)) + a;
    }
    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>> &X, std::vector<std::pair<double, double>> minmax){
        std::vector<std::vector<double>> X_norm;
        for(const auto & i : X){
            X_norm.emplace_back(s_normalize(i, minmax));
        }
        return X_norm;
    }
    std::vector<double> s_normalize(const std::vector<double> &x, std::vector<std::pair<double, double>> minmax){
        std::vector<double> x_norm;
        if(minmax.size() != x.size()) return x_norm;
        for(int j = 0; j < x.size(); j++){
            x_norm.push_back((x[j] - minmax[j].first) / (minmax[j].second - minmax[j].first));
        }
        return x_norm;
    }

    MLP::MLP(const std::vector<int> &npl) {
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

    void MLP::propagate(const std::vector<double> &inputs, bool classify) {
        for(int j = 1; j < dimensions[0]+1; j++){
            X[0][j] = inputs[j-1];
        }

        for(int l = 1; l < dimensions.size(); l++){
            for(int j = 1; j < dimensions[l]+1; j++){
                double sum = 0.0;
                for(int i = 0; i < dimensions[l-1]+1; i++){
                    sum += weights[l][i][j] * X[l-1][i];
                }
                if(l < L or classify) sum = tanh(sum);

                X[l][j] = sum;
            }
        }
    }

    std::vector<double> MLP::predict(const std::vector<double> &inputs, bool classify) {
        propagate(inputs, classify);
        std::vector<double> outputs;
        for(int j = 1; j < dimensions[L]+1; j++){
            outputs.push_back(X[L][j]);
        }
        return outputs;
    }

    void MLP::train(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, bool classify,
                    double training_rate, int epochs, bool verbose) {
        for(int e = 0; e < epochs; e++){
            if(verbose) std::cout << "Epoch " << e+1 << "/" << epochs << std::endl;

            int k = randomInt(0, X.size()-1);
            std::vector<double> x = X[k];
            std::vector<double> y = Y[k];
            propagate(x, classify);

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

            for (int l = 1; l < L+1; l++){
                for(int i = 0; i < dimensions[l-1]+1; i++){
                    for(int j = 1; j < dimensions[l]+1; j++){
                        weights[l][i][j] -= training_rate * X[l-1][i] * deltas[l][j];
                    }
                }
            }
        }
    }
}