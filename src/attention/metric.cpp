
/*
#include <torch/extension.h>
#include <torch/torch.h>

#include <torch/extension.h>


typedef torch::Tensor *tensor;
*/
#pragma once


#include <iostream>



extern "C" {
    
    int add_constant_cuda(float *a, float *b);

    int add_constant_cpp(float *a, float *b) {
        std::cout << "Hello, world!" << std::endl; 
        return add_constant_cuda(a, b);
    }
}



