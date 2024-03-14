
/*
#include <torch/extension.h>
#include <torch/torch.h>



typedef torch::Tensor *tensor;
*/


int add_constant_cuda(int a, int b);

#include <iostream> 

extern "C" {
    int add_constant_cpp(tensor self, int n) {
        std::cout << "Hello, world!" << std::endl; 
        return add_constant_cuda(self, n);
    }
}



