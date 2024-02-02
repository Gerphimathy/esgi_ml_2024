#pragma once

#include <cstdlib>
#include <vector>


namespace MachineLearning{
    std::vector<double> s_normalize(const std::vector<double> &x, std::vector<std::pair<double, double>> minmax);
    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>> &X, std::vector<std::pair<double, double>> minmax);
    double rescale(double x, double old_min, double old_max, double new_min, double new_max);
    float randomFloat(float a, float b);
    int randomInt(int a, int b);
    double distance(const std::vector<double> &x, const std::vector<double> &y);
    double mean(const std::vector<double> &x);
}
