
use tch;
use tch::nn::{self, Module };

// export LD_LIBRARY_PATH=/workspace/.pyenv_mirror/user/current/lib/python3.12/site-packages/torch/lib
// export LIBTORCH_USE_PYTORCH=1
// export RUST_BACKTRACE=full

struct ModelParameters {
    embedding_dimenson: i64,
    size_of_context_window: i64,
    size_of_vocabolary: i64,
    number_of_heads: i64
}

fn generate_init() -> nn::Init {
    nn::Init::Randn { mean: 0., stdev: 3. }
}

fn embedding_table(vs_path: &nn::Path, num_embeddings: i64, embedding_dim: i64) -> impl nn::Module {
    let config = nn::EmbeddingConfig{
        sparse: false,
        scale_grad_by_freq: false,
        ws_init: generate_init(),
        padding_idx: 0
    };
    nn::embedding(vs_path, num_embeddings, embedding_dim, config)
}


fn create_embedder_module(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module {

    let vocabolary = embedding_table(
        vs_path,
        hyper_parameters.size_of_vocabolary, 
        hyper_parameters.embedding_dimenson
    );

    let positional_encoding = embedding_table(
        vs_path, 
        hyper_parameters.size_of_context_window,
        hyper_parameters.embedding_dimenson
    );

    nn::func(move |x_bc| {
        vocabolary.forward(&x_bc) + positional_encoding.forward(&x_bc)
    })
}


fn self_attention_module(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module {

    let n = hyper_parameters.number_of_heads;
    let d = hyper_parameters.embedding_dimenson;
    let q = hyper_parameters.embedding_dimenson / hyper_parameters.number_of_heads;
    let c = hyper_parameters.size_of_context_window;

    assert!(d % n == 0, "Embeddings dimension must be divisible by the requested number of heads.");
    debug_assert_eq!(n*q, d);

    let projections_1ndq = vs_path.var("projections_1ndq", &[1, n, d, q], generate_init());
    let metric_tensors_1nqq = vs_path.var("metric_tensors_1nqq", &[1, n, q, q], generate_init());
    let mixer_1dd = vs_path.var("mixer_1dd", &[1, d, d], generate_init());

    debug_assert_eq!(projections_1ndq.size(), vec![1, n, d, q]);
    debug_assert_eq!(metric_tensors_1nqq.size(), vec![1, n, q, q]);
    debug_assert_eq!(mixer_1dd.size(), vec![1, d, d]);

    // let sqrt_q: f32 = unsafe { sqrtf32(q) };

    nn::func(move |x_bcd| {
    
        let b = x_bcd.size()[0];
        assert_eq!(x_bcd.size(), vec![b, c, d]);

        // Apply n projections to the input 
        let x_b1cd = &x_bcd.unsqueeze(1);
        let x_bncq = &x_b1cd.matmul(&projections_1ndq);
        debug_assert_eq!(x_bncq.size(), vec![b, n, c, q]);


        // Use n custom dot products to generate n score tables
        let x_bnqc = &x_bncq.transpose(-1, -2);
        let x_bncc = &x_bncq.matmul(&metric_tensors_1nqq.matmul(x_bnqc));
        debug_assert!(x_bncc.size() == vec![b, n, c, c]);
    
        // x_bnqq = &x_bnqq.divide_scalar(sqrt_q);

        let softmaxed_x_bncc = &x_bncc.softmax(-1, tch::kind::Kind::Float);
        let y_bnqc = &x_bncq.transpose(-1, -2).matmul(softmaxed_x_bncc);
        debug_assert!(y_bnqc.size() == vec![b, n, q, c]);

        let y_bcd = &y_bnqc.reshape(x_bcd.size());
        debug_assert!(y_bcd.size() == vec![b, c, d]);
    
        y_bcd.matmul(&mixer_1dd)
    })
}




fn create_layer_norm(vs_path: &nn::Path, bcd: Vec<i64>) -> impl nn::Module {
    // TODO
    let config = nn::LayerNormConfig {
        cudnn_enabled: false,
        eps: 1e-9,
        elementwise_affine: true,
        ws_init: generate_init(),
        bs_init: generate_init(),
    };
    nn::layer_norm(vs_path, bcd, config)
}


fn metric_tensor_network(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module 
{

    fn create_transformer_module(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module 
    {
        nn::seq()
        .add(create_layer_norm(vs_path))
        .add(self_attention_module(vs_path, hyper_parameters))
    }

    nn::seq()
    .add(create_embedder_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))
}




fn main() {



}


#[cfg(test)]
mod tests {
    use super::*; 
    use tch::{nn, Device, Kind, Tensor};

    fn create_hyper_parameters() -> ModelParameters {
        ModelParameters {
            number_of_heads: 16,
            size_of_context_window: 10,
            embedding_dimenson: 32,
            size_of_vocabolary: 10,
        }
    }

    fn create_model_input(batch_size: i64, hyper_parameters: &ModelParameters) -> tch::Tensor {
        Tensor::randint(
            hyper_parameters.size_of_vocabolary, 
            &[batch_size, hyper_parameters.size_of_context_window],
            (Kind::Int, Device::Cpu)
        )
    }

    fn create_latent_representation(batch_size: i64, hyper_parameters: &ModelParameters) -> tch::Tensor {
        Tensor::randn(
            &[
                batch_size,
                hyper_parameters.size_of_context_window,
                hyper_parameters.embedding_dimenson
            ], 
            (Kind::Float, Device::Cpu)
        )
    }

    #[test]
    fn test_embeding_module() {
        let vs = nn::VarStore::new(Device::Cpu);
        let hyper_parameters = create_hyper_parameters();
        let module = embedder_module(&vs.root(), &hyper_parameters);

        let b = 10; // number of batches
        let c = hyper_parameters.size_of_context_window;
        let d = hyper_parameters.embedding_dimenson;
        let input_bc = create_model_input(b, &hyper_parameters);
        let output_bcd = module.forward(&input_bc);

        assert_eq!(output_bcd.size(), vec![b, c, d]);
    }

    #[test]
    fn test_self_attention_module() {
        let vs = nn::VarStore::new(Device::Cpu);
        let hyper_parameters = create_hyper_parameters();
        let module = self_attention_module(&vs.root(), &hyper_parameters);

        let b = 10; // number of batches
        let c = hyper_parameters.size_of_context_window;
        let d = hyper_parameters.embedding_dimenson;
    
        let input_bcd = create_latent_representation(b, &hyper_parameters);
        let output_bcd = module.forward(&input_bcd);

        assert_eq!(output_bcd.size(), vec![b, c, d]);
        
    }

}
