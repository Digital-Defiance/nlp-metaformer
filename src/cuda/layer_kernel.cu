#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda.h>

#include <torch/torch.h>

using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

typedef torch::Tensor *TensorPTR;



template <typename scalar_t>
__global__ void add_tensors_kernel(scalar_t *a, scalar_t *b, scalar_t *c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] += a[i] + b[i];
}




extern "C" {
    void add_tensors_cuda(TensorPTR result, TensorPTR a, TensorPTR b) {
        AT_DISPATCH_FLOATING_TYPES(a->type(), "cuda_add_tensors", ([&] {
            add_tensors_kernel<scalar_t><<<2, 1>>>(
                a->data<scalar_t>(),
                b->data<scalar_t>(),
                result->data<scalar_t>()
            );
        }));
    }
}




class Layer : public Function<Layer> {
 public:
  static torch::Tensor forward(
      AutogradContext *ctx,
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor bias = torch::Tensor()
    ) {
        ctx->save_for_backward({input, weight, bias});
        auto output = input.mm(weight.t());
        if (bias.defined()) {
            output += bias.unsqueeze(0).expand_as(output);
        }
        return output;
  }

  static tensor_list backward(
        AutogradContext *ctx,
        tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];

        auto grad_output = grad_outputs[0];
        auto grad_input = grad_output.mm(weight);
        auto grad_weight = grad_output.t().mm(input);
        auto grad_bias = torch::Tensor();

        if (bias.defined()) {
            grad_bias = grad_output.sum(0);
        }

        return {grad_input, grad_weight, grad_bias};
  }
};
