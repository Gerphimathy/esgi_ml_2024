#pragma once
#include "ml_export.hpp"

extern "C"{
    ///Test functions
    ML_API void info();

    ///Initialization functions
    ML_API void Init();
    ML_API void Quit();

    ///MLP functions
    ML_API void delete_mlp(int index);
    ML_API void delete_all_mlps();
    ML_API int create_mlp(int* layers, unsigned int layer_count, unsigned int activation);
    ML_API bool train_mlp(int MLP, double** X, double** Y, unsigned int size,
                          bool classify, double training_rate, unsigned int epochs, int sampling, unsigned int batch_size = 0);
    ML_API double* predict_mlp(int MLP, double* X, bool classify);
    ML_API bool serialize_mlp(int MLP, const char* filename);
    ML_API int deserialize_mlp(const char* filename);

    ///Linear functions
    ML_API void delete_linear(int index);
    ML_API void delete_all_linears();
    ML_API int create_linear(int n, double b);
    ML_API bool train_linear(int linear, double** X, double* Y, unsigned int size, double learning_rate, unsigned int epochs);
    ML_API double predict_linear(int linear, double* x);
    ML_API int serialize_linear(int linear, const char* filename);
    ML_API int deserialize_linear(const char* filename);
}