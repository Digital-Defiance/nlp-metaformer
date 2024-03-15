

#include <torch/extension.h>


void t_print_vector(float *vector) {
    for (int i = 0; i < 2; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}


void t_print_all(float *a, float *b, float *c) {
        std::cout << "a: ";
        t_print_vector(a);
        std::cout << "b: ";
        t_print_vector(b);
        std::cout << "c: ";
        t_print_vector(c);
}



#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

extern "C" {

    
    void add_tensors_cuda(torch::Tensor *result, torch::Tensor *a, torch::Tensor *b);


    void add_tensors_cpp(torch::Tensor *result, torch::Tensor *a, torch::Tensor *b) {

        CHECK_CONTIGUOUS((*result));
        CHECK_CONTIGUOUS((*a));
        CHECK_CONTIGUOUS((*b));

        add_tensors_cuda(result, a, b);

    }

}

