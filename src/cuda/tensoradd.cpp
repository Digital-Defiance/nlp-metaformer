

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

extern "C" {

    void add_tensors_cuda(float *a, float *b, float *c);


    void add_tensors_cpp(torch::Tensor *result, torch::Tensor *a, torch::Tensor *b) {
        add_tensors_cuda(
            a->mutable_data_ptr<float>(),
            b->mutable_data_ptr<float>(),
            result->mutable_data_ptr<float>()
        );
    }

}

