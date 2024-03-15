#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda.h>


template <typename scalar_t>
__global__ void add_tensors_kernel(scalar_t *a, scalar_t *b, scalar_t *c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] += a[i] + b[i];
}

extern "C" {
    void add_tensors_cuda(torch::Tensor *result, torch::Tensor *a, torch::Tensor *b) {
        AT_DISPATCH_FLOATING_TYPES(a->type(), "cuda_add_tensors", ([&] {
            add_tensors_kernel<scalar_t><<<2, 1>>>(
                a->data<scalar_t>(),
                b->data<scalar_t>(),
                result->data<scalar_t>()
            );
        }));
    }
}
