#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda.h>

#include <torch/torch.h>

using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA((*x)); CHECK_CONTIGUOUS((*x))

typedef torch::Tensor *TensorPTR;

template <typename scalar_t> 
__global__ void metric_attention_forwards_kernel(
        scalar_t *input_bcd,
        scalar_t *output_bcd,
        scalar_t *metric_1nkk
) {
    /// TODO metric_attention_forwards_kernel
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    output_bcd[i] = input_bcd[i]*metric_1nkk[i]*metric_1nkk[i];
}


template <typename scalar_t> 
__global__ void metric_attention_backwards_kernel(
        scalar_t *input_bcd,
        scalar_t *metric_1nkk,

        scalar_t *grad_input_bcd,
        scalar_t *grad_metric_1nkk,
    
        scalar_t *grad_output_bcd
) {
    /// TODO metric_attention_backwards_kernel
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    grad_input_bcd[i] = grad_output_bcd[i]*metric_1nkk[i]*metric_1nkk[i];
    grad_metric_1nkk[i] = grad_output_bcd[i]*2*input_bcd[i]*metric_1nkk[i];
}


// Testing phase, this implements y_bi = w_1i*x_bi**2 for now
class MetricTensorAttention : public Function<MetricTensorAttention> {
    public:
        static torch::Tensor
        forward(
            AutogradContext *ctx,
            torch::Tensor input_bcd,
            torch::Tensor metric_1nkk
        ) {
            ctx->save_for_backward({input_bcd, metric_1nkk });

            auto device = input_bcd.device();
            auto output_bcd = torch::zeros(input_bcd.sizes()).to(device);

            AT_DISPATCH_FLOATING_TYPES(input_bcd.type(), "metric_attention_forwards_kernel", ([&] {
                metric_attention_forwards_kernel<scalar_t><<<2, 1>>>(
                    input_bcd.data<scalar_t>(),
                    output_bcd.data<scalar_t>(),
                    metric_1nkk.data<scalar_t>()
                );
            }));
            return output_bcd;
        }

        static tensor_list
        backward(
            AutogradContext *ctx,
            tensor_list grad_outputs
        ) {

            torch::Tensor grad_output_bcd = grad_outputs[0];

            auto saved = ctx->get_saved_variables();
            torch::Tensor input_bcd = saved[0];
            torch::Tensor metric_1nkk = saved[1];

            auto grad_input_bcd = torch::zeros(input_bcd.sizes()).to(input_bcd.device());
            auto grad_metric_1nkk = torch::zeros(metric_1nkk.sizes()).to(metric_1nkk.device());

            AT_DISPATCH_FLOATING_TYPES(input_bcd.type(), "metric_attention_backwards_kernel", ([&] {
                metric_attention_backwards_kernel<scalar_t><<<2, 1>>>(
                    input_bcd.data<scalar_t>(),
                    metric_1nkk.data<scalar_t>(),
                    
                    grad_input_bcd.data<scalar_t>(),
                    grad_metric_1nkk.data<scalar_t>(),
        
                    grad_output_bcd.data<scalar_t>()
                );
            }));

            return {grad_input_bcd, grad_metric_1nkk};
  }
};


extern "C" {
    void f_metric_tensor_attention(TensorPTR *out, TensorPTR input_bcd, TensorPTR metric_1nkk) {

        CHECK_INPUT(input_bcd);
        CHECK_INPUT(metric_1nkk);
        

        // taken from torch sys:
        // auto outputs__ = torch::abs(*self);
        // out__[0] = new torch::Tensor(outputs__);
        
        auto outputs = MetricTensorAttention::apply(
            *input_bcd,
            *metric_1nkk
        );
        out[0] = new torch::Tensor(outputs);
    }
}

