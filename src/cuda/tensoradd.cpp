

#include <torch/torch.h>



extern "C" {
    void add_tensors_cpp(torch::Tensor *result, torch::Tensor *a, torch::Tensor *b) {
        torch::Tensor c = a->add(*b);
        result->copy_(c);
    }
}

