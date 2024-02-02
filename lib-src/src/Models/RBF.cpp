#pragma once
#include "RBF.hpp"

namespace MachineLearning {
    pair<vector<vector<double>>, vector<vector<vector<double>>>> kmeans(const vector<vector<double>>& X, int k, int epochs){
        // Choose k random points as centroids from X
        auto centroids = vector<vector<double>>(k);
        for (int i = 0; i < k; ++i) {
            centroids[i] = X[rand() % X.size()];
        }

        bool converged = false;

        auto clusters = vector<vector<vector<double>>>(k);
        for (int i = 0; i < epochs && !converged; ++i) {
            clusters = {};
            for(auto x : X){
                vector<double> distances = {};
                for(auto c : centroids){
                    distances.push_back(distance(x, c));
                }
                clusters[min(distances.begin(), distances.end()) - distances.begin()].push_back(x);
            }

            auto old_centroids = centroids;
            centroids = {};

            for(auto cluster : clusters){
                vector<double> new_centroid(cluster[0].size(), 0);
                for(const auto& point : cluster) {
                    for(size_t i = 0; i < point.size(); ++i) {
                        new_centroid[i] += point[i];
                    }
                }
                for(auto& val : new_centroid) {
                    val /= cluster.size();
                }
                centroids.emplace_back(new_centroid);
            }

            double pattern = 0.0;
            for (int j = 0; j < k; ++j) {
                pattern += distance(centroids[j], old_centroids[j]);
            }
            if(pattern < 0.0001) converged = true;
        }

        return {centroids, clusters};
    }

    double rbf(const vector<double>& x, const vector<double>& c, double sigma){
        return exp(-pow(distance(x, c), 2) / (2 * pow(sigma, 2)));
    }

    RBF::RBF() {

    }
}