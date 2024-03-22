#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda.h>

#include <torch/torch.h>

using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA((*x)); CHECK_CONTIGUOUS((*x))

typedef torch::Tensor *TensorPTR;


template<typename scalar_t, int D>
using CudaTensorView = torch::PackedTensorAccessor32<scalar_t, D, torch::RestrictPtrTraits>;

template <typename scalar_t> 
__global__ void metric_attention_forwards_kernel(
    CudaTensorView<scalar_t, 4> p_bnck,
    CudaTensorView<scalar_t, 1> f_l, CudaTensorView<scalar_t, 1> g_l,
    CudaTensorView<scalar_t, 1> f_u, CudaTensorView<scalar_t, 1> g_u,
    CudaTensorView<scalar_t, 2> M_nl,
    CudaTensorView<scalar_t, 4> q_bnul,
    int Nb, int Nn, int Nl, int Nu 
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    int b = idx % Nb;
    idx = (idx / Nb);

    int n = idx % Nn;
    idx = idx / Nn;

    int l = idx % Nl;
    idx = idx / Nl;

    int u = idx % Nu;

    int fu = f_u[u];
    int gu = g_u[u];

    int fl = f_l[l];
    int gl = g_l[l];

    if (fl == gl and fu == gu){
        q_bnul[b][n][u][l] = M_nl[n][l]*p_bnck[b][n][fu][fl]*p_bnck[b][n][fu][fl];
    } else if (fl == gl and fu != gu) {
        q_bnul[b][n][u][l] = 2*M_nl[n][l]*p_bnck[b][n][fu][fl]*p_bnck[b][n][gu][fl];
    } else if (fl != gl and fu == gu) {
        q_bnul[b][n][u][l] = 2*M_nl[n][l]*p_bnck[b][n][fu][fl]*p_bnck[b][n][fu][gl];
    } else if (fl != gl and fu != gu) {
        q_bnul[b][n][u][l] = 4*M_nl[n][l]*p_bnck[b][n][fu][fl]*p_bnck[b][n][gu][gl];
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


class MetricTensorAttention : public Function<MetricTensorAttention> {
    public:
        static torch::Tensor
        forward(
            AutogradContext *ctx,
            torch::Tensor p_bnck,
            torch::Tensor f_l,
            torch::Tensor g_l, 
            torch::Tensor f_u,
            torch::Tensor g_u,
            torch::Tensor M_nl
        ) {
            ctx->save_for_backward({ M_nl });

            const auto device = input_bcd.device();
            auto q_nul = torch::zeros(p_nck.sizes()).to(device);

            const auto batch_size = p_nck.size(0);
            const auto Nl = f_l.size(0);
            const auto Nu = f_u.size(0);
            const auto Nn = M_nl.size(0);

            const int total_threads = batch_size*Nl*Nu*Nn
            const int threads_per_block = 1024;
            const int number_of_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

            AT_DISPATCH_FLOATING_TYPES(input_bcd.type(), "metric_attention_forwards_kernel", ([&] {
                metric_attention_forwards_kernel<scalar_t><<<number_of_blocks, threads_per_block>>>(
                    p_bnck.packed_accessor32<scalar_t, 4>(),
                    f_l.packed_accessor32<scalar_t, 1>(),
                    g_l.packed_accessor32<scalar_t, 1>(),
                    f_u.packed_accessor32<scalar_t, 1>(),
                    g_u.packed_accessor32<scalar_t, 1>(),
                    M_nl.packed_accessor32<scalar_t, 2>()
                );
            }));

            return q_bnul;
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
        TensorPTR *q_1bnu,
        TensorPTR p_bnck,
        TensorPTR f_l, TensorPTR g_l, int Nl,
        TensorPTR f_u, TensorPTR g_u, int Nu,
        TensorPTR M_nl
    ) {

        CHECK_INPUT(p_bnck);
        CHECK_INPUT(*q_1bnu);
        CHECK_INPUT(f_l); CHECK_INPUT(g_l);
        CHECK_INPUT(f_u); CHECK_INPUT(g_u);
        CHECK_INPUT(M_nl);


        q_1bnu[0] = new torch::Tensor(
            MetricTensorAttention::apply(
            *p_bnck,
            *f_l, *g_l, Nl,
            *f_u, *g_u, Nu,
            *M_nl
        ));
    }
}

