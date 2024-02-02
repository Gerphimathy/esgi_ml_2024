#pragma once
#include "../ranges.hpp"

using namespace std;

namespace MachineLearning {
    pair<vector<vector<double>>, vector<vector<vector<double>>>> kmeans(const vector<vector<double>>& X, int k, int epochs);
    double rbf(const vector<double>& x, const vector<double>& c, double sigma);
    vector<vector<double>> rbf_features(const vector<vector<double>>& X, const vector<vector<double>>& centroids, double sigma);
    class RBF {
        public:
            RBF();
    }
}
