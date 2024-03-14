#include <torch/extension.h>
#include<torch/torch.h>
#include <iostream> 


// void add_constant_cuda(float *x, float constant, int n);

typedef torch::Tensor *tensor;

extern "C" {
    void add_constant_cpp(tensor self, int c) {
            
        std::cout << "Hello, world!" << std::endl; 

        // do stuff
    }
}



