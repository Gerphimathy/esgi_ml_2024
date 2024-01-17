///Main.cpp is for testing purposes only, the C++ code is to be made into a dll

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include "Models/MLP.hpp"
#include "ranges.hpp"


struct Color {
    unsigned char r, g, b;
};

enum State {
    SOLID = 0,
    LIQUID = 1,
    GAS = 2
};

Color getStateColor(State s) {
    switch (s) {
        case SOLID:
            return {255, 0, 0};
        case LIQUID:
            return {0, 255, 0};
        case GAS:
            return {0, 0, 255};
        default:
            return {0, 0, 0};
    }
}


State getState(double T, double P) {
    ///Not the real equation, just for testing purposes
    if(P>10) return SOLID;
    else if(P < 5){
        if(T < 0) return SOLID;
        else return GAS;
    }
    else{
        if(T < 0) return SOLID;
        else if(T < 100) return LIQUID;
        else return GAS;
    }
}

///Output to a BMP file, basically our version of matplotlib
void writeBMP(const char* filename, int w, int h, const std::vector<Color>& pixels) {
    std::ofstream file(filename, std::ios::binary);

    // Bitmap File Header
    file.put('B').put('M'); // BFType
    file.put(0x36).put(0).put(0).put(0); // BFSize
    file.put(0).put(0); // BFReserved1
    file.put(0).put(0); // BFReserved2
    file.put(0x36).put(0).put(0).put(0); // BF0ffBits

    // Bitmap Info Header
    file.put(40).put(0).put(0).put(0); // Size
    file.put((char) w).put((char) (w >> 8)).put((char) (w >> 16)).put((char) (w >> 24));
    file.put((char) h).put((char) (h >> 8)).put((char) (h >> 16)).put((char) (h >> 24));
    file.put(1).put(0);
    file.put(24).put(0); // Bit count
    file.put(0).put(0).put(0).put(0);
    file.put(0).put(0).put(0).put(0); // Size raw data
    file.put((char) (0xC4)).put((char) (0x0E)).put(0).put(0);
    file.put((char) (0xC4)).put((char) (0x0E)).put(0).put(0);
    file.put(0).put(0).put(0).put(0);
    file.put(0).put(0).put(0).put(0);

    // Write in reverse order
    for (int y = h - 1; y >= 0; y--) {
        for (int x = 0; x < w; x++) {
            file.put(pixels[y * w + x].b).put(pixels[y * w + x].g).put(pixels[y * w + x].r);
        }
        if (w % 4 != 0) {
            for (int n = 0; n < 4 - w % 4; n++) {
                file.put(0);
            }
        }
    }

    file.close();
}

float randomFloat(float a, float b) {
    return ((b - a) * ((float)rand() / RAND_MAX)) + a;
}

int pixIndex(int W, int H, int w, int h, int x, int y){
    // x in range [0, w-1]
    // y in range [0, h-1]

    int X = (int) ((float)x / (w-1) * (W-1));
    int Y = (int) ((float)y / (h-1) * (H-1));

    return Y * W + X;
}

std::pair<double, double> getTP(int W, int H, int w, int h, int id){
    int x = id % W;
    int y = id / W;

    double X = (double)x / (W-1) * (w-1);
    double Y = (double)y / (H-1) * (h-1);

    return {X, Y};
}

int main() {
    int width = 300;
    int height = 300;
    int n_points = 10000;
    int epochs = 10000;
    double training_rate = 1.0/100000.0;
    MachineLearning::Activation activation = MachineLearning::SIGMOID;
    MachineLearning::Sampling sampling = MachineLearning::STOCHASTIC_GRADIANT_DESCENT;
    int batch_size = 100;
    bool verbose = true;
    bool classify = false;
    std::vector<int> layers = {2, 4, 4, 4, 1};

    ///Tests model by randomly generating training data
    /// Training Data: Simplified water phase:
    /**
     * 0: Solid
     * 1: Liquid
     * 2: Gas
     */

    srand(time(nullptr));

    /// Init output BMP
    std::vector<Color> pixels(width * height);
    for (int i = 0; i < width * height; i++) {
        pixels[i].r = 255;
        pixels[i].g = 255;
        pixels[i].b = 255;
    }

    std::vector<std::vector<double>> X(n_points);
    std::vector<std::vector<double>> Y(n_points);

    for (int i = 0; i < n_points; i++) {
        double T = randomFloat(-10, 125);
        double P = randomFloat(0,20);

        int state = getState(T, P);

        X[i] = {T, P};
        Y[i] = {(double)state/2.0};

        int id = pixIndex(width, height, 135, 20, T+10, P);
        Color c = getStateColor((State)state);
        pixels[id].r = c.r;
        pixels[id].g = c.g;
        pixels[id].b = c.b;
    }

    std::vector<std::pair<double, double>> minmax;
    minmax.emplace_back(-10, 125);
    minmax.emplace_back(0, 20);
    X = MachineLearning::normalize(X, minmax);

    MachineLearning::MLP MLP(layers, activation);

    MLP.train(X, Y, classify, training_rate, epochs, verbose, sampling, batch_size);

    for (int i = 0; i < pixels.size(); ++i) {
        Color c = pixels[i];
        if(c.r != 255 || c.g != 255 || c.b != 255) continue;

        double T, P;
        auto TP = getTP(width, height, 135, 20, i);
        T = TP.first - 10;
        P = TP.second;
        auto TPN = MachineLearning::s_normalize({T, P}, minmax);

        double pred = MLP.predict(TPN, classify)[0]*3;
        int state;
        if(pred < 1.0) state = 0;
        else if (pred < 2.0) state = 1;
        else state = 2;

        Color s = getStateColor((State)state);
        pixels[i].r = s.r/4;
        pixels[i].g = s.g/4;
        pixels[i].b = s.b/4;
    }

    writeBMP("mlp.bmp", width, height, pixels);

    return 0;
}