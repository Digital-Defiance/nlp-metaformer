

use tch::nn::{ self, Module };

use crate::metaformer::commons;
use commons::generate_init;

fn create_config() -> nn::LayerNormConfig {
    nn::LayerNormConfig {
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
    }
}

/// Enforces z-normalization across each batch independently and applies an affine transformation.
/// https://arxiv.org/abs/1607.06450
pub fn create_layer_norm(vs_path: &nn::Path, embedding_dimension: i64) -> impl Module {
    nn::layer_norm(vs_path, vec![embedding_dimension], create_config())
}