

// #include <torch/extension.h>
// #include <torch/torch.h>
//#include <torch/extension.h>
// typedef torch::Tensor *tensor;


#include <iostream>


extern "C" {
    float *add_constant_cuda(float *a, float *b);
    float *add_constant_cpp(float *a, float *b) {
        std::cout << "Hello, world!" << std::endl; 
        return add_constant_cuda(a, b);
    }
}

