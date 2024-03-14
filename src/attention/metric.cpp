#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream> 



typedef torch::Tensor *tensor;

void add_constant_cuda(tensor input, int n);


extern "C" {
    void add_constant_cpp(tensor self, int n) {
        add_constant_cuda(self, n);


        std::cout << "Hello, world!" << std::endl; 

        // do stuff
    }
}



