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
    scalar_t *p_nck,
    scalar_t *f_l, scalar_t *g_l,
    scalar_t *f_u, scalar_t *g_u,
    scalar_t *M_nl, scalar_t *q_nul
) {
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    int n = ...;
    int l = ...;
    int u = ...;

    int fu = f_u[u];
    int gu = g_u[u];

    int fl = f_l[l];
    int gl = g_l[l];

    if (fl == gl and fu == gu){
        q_nul[n][u][l] = M_nl[n][l]*p[n][fu][fl]*p[n][fu][fl];
    } else if (fl == gl and fu != gu) {
        q_nul[n][u][l] = 2*M_nl[n][l]*p[n][fu][fl]*p[n][gu][fl];
    } else if (fu == gu and fl != gl) {
        q_nul[n][u][l] = 2*M_nl[n][l]*p[n][fu][fl]*p[n][fu][gl];
    } else if (fu != gu and fl != gl) {
        q_nul[n][u][l] = 4*M_nl[n][l]*p[n][fu][fl]*p[n][gu][gl];
    }
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
            torch::Tensor p_nck,
            torch::Tensor f_l, torch::Tensor g_l, int Nl,
            torch::Tensor f_u, torch::Tensor g_u, int Nu,
            torch::Tensor M_nl
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

    // note: the naming convention relates to
    // the theoretical derivation present in the readme
    void f_metric_tensor_attention(
        TensorPTR *q_nu1,
        TensorPTR p_nck,
        TensorPTR f_l, TensorPTR g_l, int Nl,
        TensorPTR f_u, TensorPTR g_u, int Nu,
        TensorPTR M_nl
    ) {

        CHECK_INPUT(p_nck);
        CHECK_INPUT(f_l); CHECK_INPUT(g_l);
        CHECK_INPUT(f_u); CHECK_INPUT(g_u);
        CHECK_INPUT(M_nl);
        
        auto outputs = MetricTensorAttention::apply(
            *p_nck,
            *f_l, *g_l, Nl,
            *f_u, *g_u, Nu,
            *M_nl
        );

        q_nu1[0] = new torch::Tensor(outputs);
    }
}

