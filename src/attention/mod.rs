pub mod quadratic_form;
pub mod scaled_dot_product;

use crate::attention::quadratic_form::QuadraticAttention;
use crate::attention::scaled_dot_product::ScaledDotProductAttention;

use tch::Tensor;
use tch::nn::Module;


#[derive(Debug)]
pub(crate) enum AttentionModule {
    Quadratic(QuadraticAttention),
    ScaledDotProduct(ScaledDotProductAttention),
    // Other variants
}

impl Module for AttentionModule {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            AttentionModule::Quadratic(module) => module.forward(xs),
            AttentionModule::ScaledDotProduct(module) => module.forward(xs),
        }
    }
}

