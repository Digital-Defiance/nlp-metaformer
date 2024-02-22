/*

    TODO: residual connections
    TODO: output tokenizer
    TODO: the 1/sqrt(q) scale before the softmax in the self attention module
*/
use tch;
use tch::nn::{self, Module };


/*
export LD_LIBRARY_PATH=/workspace/.pyenv_mirror/user/current/lib/python3.12/site-packages/torch/lib
export LIBTORCH_USE_PYTORCH=1
export RUST_BACKTRACE=full
*/



/// Defines structure of the quadratic attention model
pub struct ModelParameters {

    /// Dimension of the vector space that the network
    /// uses internally to represent tokens 
    embedding_dimenson: i64,

    /// Maximum number of tokens in the input sequence
    size_of_context_window: i64,

    /// Total number of tokens that the network recognizes
    size_of_vocabolary: i64,

    /// Number of attention modules per transformer block
    number_of_heads: i64
}

fn generate_init() -> nn::Init {
    nn::Init::Randn { mean: 0., stdev: 1. }
}


/// Transforms integer valued tokens into positionally encoded embeddings.
fn create_embedder_module(vs: &nn::Path, hp: &ModelParameters) -> impl nn::Module {
 
    let config: nn::EmbeddingConfig = nn::EmbeddingConfig{
    
        /*
        If True, gradient w.r.t. weight matrix will be a sparse tensor.
        See Notes for more details regarding sparse gradients
         */
        sparse: false,

        // If given, this will scale gradients by the inverse of frequency of the words in the mini-batch
        scale_grad_by_freq: false,

        /*
        If specified, the entries at padding_idx do not contribute to the gradient; 
        therefore, the embedding vector at padding_idx is not updated during training, 
        i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the embedding 
        vector at padding_idx will default to all zeros, but can be updated to another
        value to be used as the padding vector.
         */
        padding_idx: 0,

        ws_init: generate_init(),
    };

    let d: i64 = hp.embedding_dimenson;
    let v: i64 = hp.size_of_vocabolary;
    let c: i64 = hp.size_of_context_window;

    let vocabolary_vd = nn::embedding(vs, v, d, config);
    let positional_encoding_cd = nn::embedding(vs, c, d, config);

    nn::func(move |x_bc: &tch::Tensor| {
        vocabolary_vd.forward(&x_bc) + positional_encoding_cd.forward(&x_bc)
    })
}

/// Performs self attention N times using the quadratic form $xW_nx.T$ where $W_n$ is a learnable matrix.
/// This is an early version of the metric self attention, where $W$ is forced to have the properties a metric tensor.
fn quadratic_self_attention_module(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module {

    let n: i64 = hyper_parameters.number_of_heads;
    let d: i64 = hyper_parameters.embedding_dimenson;
    let q: i64 = hyper_parameters.embedding_dimenson / hyper_parameters.number_of_heads;
    let c: i64 = hyper_parameters.size_of_context_window;

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



/// Enforces z-normalization across each batch independently and applies an affine transformation.
fn create_layer_norm(vs_path: &nn::Path, embedding_dimension: i64) -> impl nn::Module {

    let config: nn::LayerNormConfig = nn::LayerNormConfig {
        /*a value added to the denominator for numerical stability. Default: 1e-5 */
        eps: 1e-5,
    
        /*
        a boolean value that when set to True, this module has learnable
        per-element affine parameters initialized to ones (for weights)
        and zeros (for biases).
         */
        elementwise_affine: true,
    
        ws_init: generate_init(),
        bs_init: generate_init(),
        cudnn_enabled: false,
    };

    nn::layer_norm(vs_path, vec![embedding_dimension], config)
}



/// Dense feed forward with GELU activation
fn mlp_module(vs: &nn::Path, embedding_dimension: i64) -> impl nn::Module {

    let d: i64 = embedding_dimension;
    let q: i64 = embedding_dimension / 2;

    let projection_1dq = vs.var("projection_1dq", &[1, d, q], generate_init());
    let expansion_1qd = vs.var("expansion_1qd", &[1, q, d], generate_init());

    nn::func(move |x_bcd: &tch::Tensor| {
        let x_bcq = &x_bcd.matmul(&projection_1dq);
        let activations_bcq = x_bcq.gelu("none");
        activations_bcq.matmul(&expansion_1qd)
    })
}


fn create_transformer_module(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module 
{
    nn::seq()
    .add(create_layer_norm(vs_path, hyper_parameters.embedding_dimenson))
    .add(quadratic_self_attention_module(vs_path, hyper_parameters))
    .add(create_layer_norm(vs_path, hyper_parameters.embedding_dimenson))
    .add(mlp_module(vs_path, hyper_parameters.embedding_dimenson))
}


pub fn quadratic_tensor_network(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module 
{



    nn::seq()
    .add(create_embedder_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))
    .add(create_transformer_module(vs_path, hyper_parameters))

    // .add(create_output_tokenizer(vs_path, hyper_parameters.output_tokens))
}




fn main() {



}





macro_rules! generate_test {
    ($test_name:ident, $module_func:expr, $input_factory:expr) => {
        #[test]
        fn $test_name() {
            let vs = nn::VarStore::new(Device::Cpu);
            let hyper_parameters = create_hyper_parameters();
            let module = $module_func(&vs.root(), &hyper_parameters);

            let batch_size = 10; // Assuming batch size is constant for simplicity
            let input = $input_factory(batch_size, &hyper_parameters);

            let output = module.forward(&input);

            let expected_size = vec![batch_size, hyper_parameters.size_of_context_window, hyper_parameters.embedding_dimenson];
            assert_eq!(output.size(), expected_size, "Failed test: {:?}", stringify!($test_name));
        }
    };
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

    generate_test!(test_embedding_module, create_embedder_module, create_model_input);

    generate_test!(test_transformer_module, create_transformer_module, create_latent_representation);
    generate_test!(test_quadratic_attention_module, quadratic_self_attention_module, create_latent_representation);

    generate_test!(test_trasnformer_block, create_transformer_module, create_latent_representation);

    generate_test!(
        test_mlp_module,
        |vs_path, hyper_parameters: &ModelParameters| mlp_module(vs_path, hyper_parameters.embedding_dimenson),
        create_latent_representation
    );


    generate_test!(test_quadratic_tensor_network, quadratic_tensor_network, create_model_input);




}
