

// #include <torch/extension.h>
// #include <torch/torch.h>
//#include <torch/extension.h>
// typedef torch::Tensor *tensor;


#include <iostream>


extern "C" {
    
    /// @brief Forward declaration of the wrapper around the CUDA kernel.
    /// @param a TODO
    /// @param b TODO
    /// @return TODO
    float *add_constant_cuda(float *a, float *b);

    /// @brief Adds two tensors toguether.
    /// @param a TODO
    /// @param b TODO
    /// @return TODO
    float *add_constant_cpp(float *a, float *b) {
        std::cout << "Hello, world!" << std::endl; 
        return add_constant_cuda(a, b);
    }
}



