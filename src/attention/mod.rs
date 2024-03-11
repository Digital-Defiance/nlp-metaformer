pub mod quadratic_form;
pub mod scaled_dot_product;
pub mod identity;
pub mod avg_pooling;

use crate::attention::quadratic_form::QuadraticAttention;
use crate::attention::scaled_dot_product::ScaledDotProductAttention;
use crate::attention::identity::Identity;

use tch::Tensor;
use tch::nn::Module;


#[derive(Debug)]
pub(crate) enum AttentionModule {
    Identity(Identity),
    QuadraticAttention(QuadraticAttention),
    ScaledDotProduct(ScaledDotProductAttention),
}

impl Module for AttentionModule {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            AttentionModule::Identity(module) => module.forward(xs),
            AttentionModule::QuadraticAttention(module) => module.forward(xs),
            AttentionModule::ScaledDotProduct(module) => module.forward(xs),

        }
    }
}

