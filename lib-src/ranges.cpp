#include "ranges.hpp"

namespace MachineLearning{
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
            if(minmax[j].first == minmax[j].second) x_norm.push_back(0.0);
            else x_norm.push_back((x[j] - minmax[j].first) / (minmax[j].second - minmax[j].first));
        }
        return x_norm;
    }

    double rescale(double x, double old_min, double old_max, double new_min, double new_max){
        if(old_max == old_min) return 0.0;
        return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min;
    }
}


