#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda.h>

#include <torch/torch.h>

using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA((*x)); CHECK_CONTIGUOUS((*x))

template<typename scalar_t, size_t D>
using CudaTensorView = torch::PackedTensorAccessor32<scalar_t, D, torch::RestrictPtrTraits>;
using constants_list = std::vector<at::Tensor>;


template <typename scalar_t> 
__global__ void metric_attention_forwards_kernel(
    CudaTensorView<scalar_t, 4> p_bnck,
    CudaTensorView<scalar_t, 2> M_nl,
    CudaTensorView<scalar_t, 4> q_bnul,
    CudaTensorView<size_t, 2> index_table_2l,
    CudaTensorView<size_t, 2> index_table_2u
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    // TODO: index wizardy
    size_t b = ;
    size_t u = ;
    size_t l = ;

    size_t k = ...;
    size_t k_1 = ...;

    size_t c = ...;
    size_t c_1 = ...;

    // assign common factor
    q_bnul[b][n][u][l] =  M_nl[n][l]*p_bnck[b][n][c][k];

    if (k == k_1 && c == c_1){
        q_bnul[b][n][u][l] *= p_bnck[b][n][c][k];
    } else if (k == k_1  && c != c_1) {
        q_bnul[b][n][u][l] *= 2*p_bnck[b][n][c_1][k];
    } else if (k != k_1  && c == c_1) {
        q_bnul[b][n][u][l] *= 2*p_bnck[b][n][c][k_1];
    } else if (k != k_1  && c != c_1) {
        q_bnul[b][n][u][l] *= 4*p_bnck[b][n][c_1][k_1];
    }
}


template <typename scalar_t>
__global__ void metric_attention_backwards_kernel_p(
    CudaTensorView<scalar_t, 4> p_bnck,
    CudaTensorView<scalar_t, 2> M_nl,
    CudaTensorView<scalar_t, 6> grad_r_bnul__p_bnck,
    CudaTensorView<size_t, 2> index_table_2l,
    CudaTensorView<size_t, 2> index_table_2u
) {

    // TODO: index wizzardy 
    size_t c_2 = ...;
    size_t k_2 = ...;

    if (c_2 == c  && k_2 == k){
        grad_r_bnul__p_bnck[b][n][u][l][c_2][k_2] += M_nl[n][l]*p_bnck[b][n][c_1][k_1];
    }

    if (c_2 == c_1  && k_2 == k_1){
        grad_r_bnul__p_bnck[b][n][u][l][c_2][k_2] += M_nl[n][l]*p_bnck[b][n][c][k];
    }
}




template <typename scalar_t> 
__global__ void metric_attention_backwards_kernel_M(
    CudaTensorView<scalar_t, 4> p_bnck,
    CudaTensorView<scalar_t, 2> M_nl,
    CudaTensorView<scalar_t, 4> grad_r_bnu__M_nl,
    CudaTensorView<scalar_t, 4> q_bnul,
    CudaTensorView<size_t, 2> index_table_2l,
    CudaTensorView<size_t, 2> index_table_2u
) {
    // TODO: index wizzardy 
    size_t c_2 = ...;
    size_t k_2 = ...;

    grad_r_bnu__M_nl[b][n][u][l] = q_bnul[b][n][u][l] / M_nl[n][l];
}


class MetricTensorAttention : public Function<MetricTensorAttention> {
    public:
        static variable_list forward(
            AutogradContext *ctx,
            Variable p_bnck,
            Variable M_nl,
            constants_list index_tables
        ) {

            const auto device = p_bnck.device();
    
            auto q_nul = torch::zeros(p_bnck.sizes()).to(device);
            auto index_table_2l = index_tables[0];
            auto index_table_2u = index_tables[1];
      
            const auto Nb = p_bnck.size(0);
            const auto Nl = M_nl.size(1);
            const auto Nn = M_nl.size(0);
            const auto Nu = f_u.size(0);

            const int total_threads = Nb*Nl*Nu*Nn;
            const int threads_per_block = 1024;
            const int number_of_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

            auto r_bnul = torch::zeros((Nb, Nn, Nu, Nl));

            AT_DISPATCH_FLOATING_TYPES(p_bnck.type(), "metric_attention_forwards_kernel", ([&] {
                metric_attention_forwards_kernel<scalar_t><<<number_of_blocks, threads_per_block>>>(
                    p_bnck.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    M_nl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    r_bnul.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    index_table_2l.packed_accessor32<size_t, 2, torch::RestrictPtrTraits>(),
                    index_table_2u.packed_accessor32<size_t, 2, torch::RestrictPtrTraits>()
                );
            }));

            ctx->save_for_backward({  p_bnck, M_nl, index_table_l, index_table_u  });

            auto r_bnu = r_bnul.sum(dim=-1);

            return { r_bnu };
        }


        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {

            auto grad_network__r_bnul = grad_outputs[0];
            const auto device = grad_r_bnul.device();

            auto saved = ctx->get_saved_variables();

            auto p_bnck = saved[0];
            auto M_nl = saved[1];
            auto index_table_2l = saved[2];
            auto index_table_2u = saved[3];
        

            auto grad_r_bnul__p_bnck = torch::zeros(
                (b, n, u, l, c, k)
            ).to(device);

            AT_DISPATCH_FLOATING_TYPES(input_bcd.type(), "metric_attention_backwards_kernel_p", ([&] {
                metric_attention_backwards_kernel_p<scalar_t><<<2, 1>>>(
                    p_bnck.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    M_nl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_r_bnul__p_bnck.packed_accessor32<scalar_t, 6, torch::RestrictPtrTraits>(),
                    grad_r_bnu__M_nl.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    index_table_2l.packed_accessor32<size_t, 2, torch::RestrictPtrTraits>()>,
                    index_table_2u.packed_accessor32<size_t, 2, torch::RestrictPtrTraits>()>
                    
                );
            }));


            auto grad_r_bnu__M_nl  = torch::zeros(
                (b, n, u, l)
            ).to(device);

            AT_DISPATCH_FLOATING_TYPES(input_bcd.type(), "metric_attention_backwards_kernel_p", ([&] {
                metric_attention_backwards_kernel_p<scalar_t><<<2, 1>>>(
                    p_bnck.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    M_nl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_r_bnul__p_bnck.packed_accessor32<scalar_t, 6, torch::RestrictPtrTraits>(),
                    grad_r_bnu__M_nl.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    index_table_2l.packed_accessor32<size_t, 2, torch::RestrictPtrTraits>()>,
                    index_table_2u.packed_accessor32<size_t, 2, torch::RestrictPtrTraits>()>
                    
                );
            }));


            // TODO: chain rule, grad_network__r_bnul
            return { grad_r_bnul__p_bnck, grad_r_bnu__M_nl };
  }
};


typedef torch::Tensor *TensorPTR;


extern "C" {

    // note: the naming convention relates to
    // the theoretical derivation present in the readme
    void f_metric_tensor_attention(
        TensorPTR *q_1bnu,
        TensorPTR p_bnck,
        TensorPTR M_nl,
        TensorPTR index_table_l,
        TensorPTR index_table_u,
    ) {

        CHECK_INPUT(p_bnck);
        CHECK_INPUT(*q_1bnu);
        CHECK_INPUT(index_table_l);
        CHECK_INPUT(index_table_u);
        CHECK_INPUT(M_nl);

        constants_list index_tables = {*index_table_l, *index_table_u };
        auto res = MetricTensorAttention::apply(
                *p_bnck,
                *M_nl,
                index_tables
        )[0];
        q_1bnu[0] = new torch::Tensor(res);
    }
}

