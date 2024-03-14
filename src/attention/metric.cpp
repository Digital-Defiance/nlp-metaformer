#include <torch/extension.h>

// void add_constant_cuda(float *x, float constant, int n); // Declare the CUDA function


extern "C" {
    int add_constant_cpp(int x, int c) {
        return x + c ;
    }
}



