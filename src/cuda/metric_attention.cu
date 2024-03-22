#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda.h>

#include <torch/torch.h>

using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA((*x)); CHECK_CONTIGUOUS((*x))

typedef torch::Tensor *TensorPTR;
typedef const int Vec1[];

template<typename scalar_t, size_t D>
using CudaTensorView = torch::PackedTensorAccessor32<scalar_t, D, torch::RestrictPtrTraits>;

template <typename scalar_t> 
__global__ void metric_attention_forwards_kernel(
    CudaTensorView<scalar_t, 4> p_bnck,
    Vec1 f_l, Vec1 g_l, Vec1 f_u, Vec1 g_u,
    CudaTensorView<scalar_t, 2> M_nl,
    CudaTensorView<scalar_t, 4> q_bnul
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    
    int Nb = q_bnul.size(0);
    int b = idx % Nb;
    idx = (idx / Nb);
    
    int Nn = q_bnul.size(1);
    int n = idx % Nn;
    idx = idx / Nn;

    int Nl = q_bnul.size(3);
    int l = idx % Nl;
    idx = idx / Nl;

    int Nl = q_bnul.size(2);
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
    CudaTensorView<scalar_t, 4> p_bnck,
    Vec1 f_l, Vec1 g_l, Vec1 f_u, Vec1 g_u,
    CudaTensorView<scalar_t, 2> M_nl,
    CudaTensorView<scalar_t, 4> q_bnul
) {
    /// TODO metric_attention_backwards_kernel
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    grad_input_bcd[i] = grad_output_bcd[i]*metric_1nkk[i]*metric_1nkk[i];
    grad_metric_1nkk[i] = grad_output_bcd[i]*2*input_bcd[i]*metric_1nkk[i];
}


class MetricTensorAttention : public Function<MetricTensorAttention> {
    public:
        static variable_list forward(
            AutogradContext *ctx,
            Variable p_bnck,
            Vec1 f_l,
            Vec1 g_l,
            Vec1 f_u,
            Vec1 g_u,
            const int Nu,
            Variable M_nl
        ) {
            ctx->save_for_backward({ M_nl });

            const auto device = p_bnck.device();
            auto q_nul = torch::zeros(p_bnck.sizes()).to(device);

            const auto Nb = p_bnck.size(0);
            const auto Nl =  M_nl.size(1);
            const auto Nn = M_nl.size(0);

            const int total_threads = Nb*Nl*Nu*Nn;
            const int threads_per_block = 1024;
            const int number_of_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
            
            auto q_bnul = torch::zeros((Nb, Nn, Nu, Nl));

            AT_DISPATCH_FLOATING_TYPES(p_bnck.type(), "metric_attention_forwards_kernel", ([&] {
                metric_attention_forwards_kernel<scalar_t><<<number_of_blocks, threads_per_block>>>(
                    p_bnck.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    f_l, g_l, f_u, g_u,
                    M_nl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    q_bnul.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
                );
            }));

            return { q_bnul };
        }

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {

            torch::Tensor grad_q_bnul = grad_outputs[0];

            auto saved = ctx->get_saved_variables();
            auto p_bnck = saved[0];
            auto M_nl = saved[1];
          
            const auto device = M_nl.device();
            auto grad_p_bnck = torch::zeros_like(p_bnck).to(device);
            auto grad_M_nl  = torch::zeros_like(M_nl).to(device);

            AT_DISPATCH_FLOATING_TYPES(input_bcd.type(), "metric_attention_backwards_kernel", ([&] {
                metric_attention_backwards_kernel<scalar_t><<<2, 1>>>(
                    grad_q_bnul.data<scalar_t>(),

                    p_bnck.data<scalar_t>(),                    
                    
                    M_nl.data<scalar_t>(),
                    grad_p_bnckd.data<scalar_t>(),
                    grad_M_nl.data<scalar_t>()
                );
            }));

            return {grad_p_bnck, grad_M_nl};
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

