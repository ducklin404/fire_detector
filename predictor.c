#include <math.h>    
#include <stdbool.h>
#include "model.h"    

#define N_FEATURES 5

// prob of fire
float predict_probability(const float input[N_FEATURES]) {
    float z = lr_intercept;
    for (int i = 0; i < N_FEATURES; ++i) {

        float x_scaled = (input[i] - scaler_mean[i]) / scaler_scale[i];
        z += lr_coeffs[i] * x_scaled;
    }
    // sigmoid
    float prob = 1.0f / (1.0f + expf(-z));
    return prob;
}

