#include "ext.h"
#include <iostream>
#include "Models/MLP.hpp"

///Test functions
void info(){
    std::cout << "Test" << std::endl;
}

///Initialization functions
void Init(){
    srand(time(nullptr));
}

void Quit(){
    delete_all_mlps();
}

///MLP functions
static std::vector<MachineLearning::MLP> MLPs;

void delete_mlp(int index){
    if(index < 0 || index >= MLPs.size()) return;
    MLPs[index].~MLP();
    MLPs.erase(MLPs.begin() + index);
}
void delete_all_mlps(){
    for(auto& mlp : MLPs){
        mlp.~MLP();
    }
}

int create_mlp(int* layers, unsigned int layer_count, unsigned int activation){
    if(layer_count < 2) return -1;
    MachineLearning::Activation a = MachineLearning::get_activation(activation);

    std::vector<int> dims;

    for(unsigned int i = 0; i < layer_count; i++){
        if(layers[i] < 1) return -1;
        dims.push_back(layers[i]);
    }

    MLPs.emplace_back(dims, a);
    return MLPs.size() - 1;
}

bool train_mlp(int MLP, double** X, double** Y, unsigned int size,
               bool classify, double training_rate, unsigned int epochs, int sampling, unsigned int batch_size){
    if(MLP < 0 || MLP >= MLPs.size()) return false;

    MachineLearning::Sampling s = MachineLearning::get_sampling(sampling);

    if(s == MachineLearning::Sampling::MINI_BATCH_GRADIANT_DESCENT && (batch_size <= 0 || batch_size > size)) return false;

    std::vector<std::vector<double>> x = std::vector<std::vector<double>>(size);
    std::vector<std::vector<double>> y = std::vector<std::vector<double>>(size);

    auto dims = MLPs[MLP].get_dimensions();

    int x_size = dims[0];
    int y_size = dims[dims.size() - 1];

    try{
        for (int i = 0; i < size; ++i) {
            x[i] = std::vector<double>(x_size);
            y[i] = std::vector<double>(y_size);
            //TODO: Pretty sure there's a way to do this through copying memory
            for (int j = 0; j < x_size; ++j) x[i][j] = X[i][j];
            for (int j = 0; j < y_size; ++j) y[i][j] = Y[i][j];
        }
    } catch (std::exception& e){
        return false;
    }

    MLPs[MLP].train(x, y, classify, training_rate, epochs, false, s, batch_size);
    return true;
}

double* predict_mlp(int MLP, double* X, bool classify){
    if(MLP < 0 || MLP >= MLPs.size()) return nullptr;

    auto dims = MLPs[MLP].get_dimensions();

    int x_size = dims[0];
    int y_size = dims[dims.size() - 1];

    std::vector<double> x = std::vector<double>(x_size);
    std::vector<double> y = std::vector<double>(y_size);

    //TODO: Memcopy ?

    for (int j = 0; j < x_size; ++j) x[j] = X[j];

    y = MLPs[MLP].predict(x, classify);

    double* ret = new double[y_size];

    for (int j = 0; j < y_size; ++j) ret[j] = y[j];

    return ret;
}

bool serialize_mlp(int MLP, const char* filename){
    if(MLP < 0 || MLP >= MLPs.size()) return false;
    MLPs[MLP].serialize(filename);
    return true;
}

int deserialize_mlp(const char* filename){
    MLPs.emplace_back(filename);
    return MLPs.size() - 1;
}