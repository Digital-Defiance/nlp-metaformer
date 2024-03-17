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
        scalar_t *x_bcd,
        scalar_t *metric_1nkk,
        scalar_t *result_bcd
) {
    /// TODO metric_attention_forwards_kernel
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // c[i] += a[i] + b[i];
}


template <typename scalar_t> 
__global__ void metric_attention_backwards_kernel(
        scalar_t *x_bcd,
        scalar_t *metric_1nkk,
        scalar_t *result_bcd
) {
    /// TODO metric_attention_backwards_kernel
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // c[i] += a[i] + b[i];
}


class MetricTensorAttention : public Function<MetricTensorAttention> {
    public:
        static void
        forward(
            AutogradContext *ctx,
            TensorPTR x_bcd,
            TensorPTR metric_1nkk,
            TensorPTR result_bcd,
            int b, int c, int d, int n, int k
        ) {
            ctx->save_for_backward({x_bcd, metric_1nkk, result_bcd});

            AT_DISPATCH_FLOATING_TYPES(x_bcd->type(), "metric_attention_forwards_kernel", ([&] {
                metric_attention_backwards_kernel<scalar_t><<<2, 1>>>(
                    x_bcd->data<scalar_t>(),
                    metric_1nkk->data<scalar_t>(),
                    result->data<scalar_t>()
                );
            });
        }

        static tensor_list
        backward(
            AutogradContext *ctx,
            tensor_list grad_outputs
        ) {
            auto saved = ctx->get_saved_variables();
            // x_bcd, metric_1nkk, result_bcd
            auto x_bcd = saved[0];
            auto metric_1nkk = saved[1];
            auto result_bcd = saved[2];

            AT_DISPATCH_FLOATING_TYPES(x_bcd->type(), "metric_attention_backwards_kernel", ([&] {
                metric_attention_backwards_kernel<scalar_t><<<2, 1>>>(
                    x_bcd->data<scalar_t>(),
                    metric_1nkk->data<scalar_t>(),
                    result->data<scalar_t>()
                );
            });  
                  
            // auto grad_output = grad_outputs[0];
            // auto grad_input = grad_output.mm(weight);
            // auto grad_weight = grad_output.t().mm(input);

            return {}; // {grad_input, grad_weight};
  }
};


extern "C" {
    void f_metric_tensor_attention(TensorPTR x_bcd, TensorPTR result_bcd, TensorPTR metric_1nkk) {

        CHECK_INPUT(x_bcd);
        CHECK_INPUT(result_bcd);
        CHECK_INPUT(metric_1nkk);

        torch::Tensor


        MetricTensorAttention::apply(
            x_bcd,
            metric_1nkk
        );
    }
}

