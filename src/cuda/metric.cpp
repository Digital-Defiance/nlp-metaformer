

// #include <torch/extension.h>
// #include <torch/torch.h>
//#include <torch/extension.h>
// typedef torch::Tensor *tensor;


#include <iostream>


extern "C" {
    float *add_vectors_cuda(float *a, float *b, int n);
    float *add_vectors_cpp(float *a, float *b, int n) {
        std::cout << "Hello, world!" << std::endl; 

        return add_vectors_cuda(a, b, n);
    }
}

