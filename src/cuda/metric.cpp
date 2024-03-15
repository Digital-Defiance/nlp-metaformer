

// #include <torch/extension.h>
// #include <torch/torch.h>
//#include <torch/extension.h>
// typedef torch::Tensor *tensor;


#include <iostream>


extern "C" {
    void add_vectors_cuda(float *result, float *a, float *b, int n);
    float *add_vectors_cpp(float *a, float *b, int n) {
        std::cout << "Hello, world from cpp!" << std::endl; 
        float *result;
        add_vectors_cuda(result, a, b, n);
        return result;
    }
}

