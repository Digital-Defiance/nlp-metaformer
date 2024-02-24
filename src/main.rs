/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional

    TODO: output tokenizer


export LD_LIBRARY_PATH=/workspace/.pyenv_mirror/user/current/lib/python3.12/site-packages/torch/lib
export LIBTORCH_USE_PYTORCH=1
export RUST_BACKTRACE=full
*/
use tch;
use tch::nn::{ self, Module };

pub mod metaformer;
pub mod attention;


use metaformer::commons::ModelParameters;





/// Implementation of gradient descent
/// https://paperswithcode.com/method/adam
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
            model_depth: 6,
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
