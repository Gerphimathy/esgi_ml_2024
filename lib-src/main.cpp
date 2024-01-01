///Main.cpp is for testing purposes only, the C++ code is to be made into a dll

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include "Models/Linear.hpp"
#include "Models/MLP.hpp"


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

std::vector<double> stateHotEncode(State s) {
    switch (s) {
        case SOLID:
            return {1, 0, 0};
        case LIQUID:
            return {0, 1, 0};
        case GAS:
            return {0, 0, 1};
        default:
            return {0, 0, 0};
    }
}

int stateHotDecode(std::vector<double> s) {
    // Return max index
    int max_i = 0;
    for (int i = 1; i < s.size(); i++) {
        if (s[i] > s[max_i]) max_i = i;
    }
    return max_i;
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
    int epochs = 1000;
    int epochs2 = 500000;
    double training_rate = 1.0/10000.0;
    double training_rate2 = 1.0/500000.0;

    /// Init output BMP
    std::vector<Color> pixels(width * height);
    std::vector<Color> pixels2(width * height);
    for (int i = 0; i < width * height; i++) {
        pixels[i].r = 255;
        pixels[i].g = 255;
        pixels[i].b = 255;

        pixels2[i].r = 255;
        pixels2[i].g = 255;
        pixels2[i].b = 255;
    }

    srand(time(nullptr));

    ///Randomly generate training data
    /// Training Data: Simplified water phase:
    /**
     * 0: Solid
     * 1: Liquid
     * 2: Gas
     */
    std::vector<std::vector<double>> X(n_points);
    std::vector<double> Y(n_points);
    std::vector<std::vector<double>> X_Multi(n_points);
    std::vector<std::vector<double>> Y_Multi(n_points);

    for (int i = 0; i < n_points; i++) {
        double T = randomFloat(-10, 125);
        double P = randomFloat(0,20);

        ///Not the real equation, just for testing purposes
        int state = getState(T, P);

        X[i] = {T, P};
        Y[i] = state;
        Y_Multi[i] = stateHotEncode((State)state);

        //Test value generation
        int id = pixIndex(width, height, 135, 20, T+10, P);
        Color c = getStateColor((State)state);
        pixels[id].r = c.r;
        pixels[id].g = c.g;
        pixels[id].b = c.b;

        pixels2[id].r = c.r;
        pixels2[id].g = c.g;
        pixels2[id].b = c.b;
    }

    std::vector<std::pair<double, double>> minmax;
    minmax.emplace_back(-10, 125);
    minmax.emplace_back(0, 20);
    X_Multi = MachineLearning::normalize(X, minmax);

    MachineLearning::Linear Linear(2, 0);
    Linear.train(X, Y, training_rate, epochs);

    for (int i = 0; i < pixels.size(); ++i) {
        Color c = pixels[i];
        if(c.r != 255 || c.g != 255 || c.b != 255) continue;

        double T, P;
        auto TP = getTP(width, height, 135, 20, i);
        T = TP.first - 10;
        P = TP.second;

        int pred = round(Linear.predict({T, P}));
        if(pred < 0) pred = 0;
        else if(pred > 2) pred = 2;

        Color s = getStateColor((State)pred);
        pixels[i].r = s.r/4;
        pixels[i].g = s.g/4;
        pixels[i].b = s.b/4;
    }

    writeBMP("linear.bmp", width, height, pixels);

    std::cout << "Linear finished" << std::endl;

    MachineLearning::MLP MLP({2, 5, 3});

    std::cout << "Testing MLP" << std::endl;

    MLP.train(X_Multi, Y_Multi, true, training_rate2, epochs2);

    for (int i = 0; i < pixels.size(); ++i) {
        Color c = pixels2[i];
        if(c.r != 255 || c.g != 255 || c.b != 255) continue;

        double T, P;
        auto TP = getTP(width, height, 135, 20, i);
        T = TP.first - 10;
        P = TP.second;

        auto pred = MLP.predict({T, P}, true);
        std::pair<double, double> minmax2 = {-1, 1};
        pred = MachineLearning::s_normalize(pred, {minmax2, minmax2, minmax2});

        pixels2[i].r = pred[0]*255;
        pixels2[i].g = pred[1]*255;
        pixels2[i].b = pred[2]*255;
    }

    writeBMP("mlp.bmp", width, height, pixels2);

    return 0;
}