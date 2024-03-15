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
__global__ void metric_attention_kernel(scalar_t *x_bcd, scalar_t *metric_1nkk) {
    
    /// TO DO 
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;


    c[i] += a[i] + b[i];
}

class MetricTensorAttention : public Function<MetricTensorAttention> {
    public:
        static torch::Tensor
        forward(
            AutogradContext *ctx,
            torch::Tensor x_bcd,
            torch::Tensor metric_1nkk
        ) {
            ctx->save_for_backward({x_bcd, metric_1nkk});
            auto output_bcd = x_bcd.mm(metric_1nkk.t());

            // TO DO
            /*
            
            
            AT_DISPATCH_FLOATING_TYPES(a->type(), "cuda_add_tensors", ([&] {
            add_tensors_kernel<scalar_t><<<2, 1>>>(
                a->data<scalar_t>(),
                b->data<scalar_t>(),
                result->data<scalar_t>()
            );
        }));
            */

            return output_bcd;
        }

        static tensor_list
        backward(
            AutogradContext *ctx,
            tensor_list grad_outputs
        ) {
            auto saved = ctx->get_saved_variables();
            auto input = saved[0];
            auto weight = saved[1];
            auto bias = saved[2];


            // TO DO
            /*
            
            
            AT_DISPATCH_FLOATING_TYPES(a->type(), "cuda_add_tensors", ([&] {
            add_tensors_kernel<scalar_t><<<2, 1>>>(
                a->data<scalar_t>(),
                b->data<scalar_t>(),
                result->data<scalar_t>()
            );
        }));
            */
        
            auto grad_output = grad_outputs[0];
            auto grad_input = grad_output.mm(weight);
            auto grad_weight = grad_output.t().mm(input);

            return {grad_input, grad_weight};
  }
};


extern "C" {
    void f_metric_tensor_attention(TensorPTR x_bcd, TensorPTR metric_1nkk) {
        MetricTensorAttention::apply(
            *x_bcd,
            *metric_1nkk
        );
    }
}

