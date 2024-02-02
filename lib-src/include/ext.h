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
}