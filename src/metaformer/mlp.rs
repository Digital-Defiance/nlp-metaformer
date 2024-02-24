use tch::nn::{self, Module};

use crate::metaformer::commons::generate_init;
use crate::metaformer::layer_norm::create_layer_norm;




/// Dense feed forward with GELU activation
/// https://arxiv.org/abs/2202.05262
pub fn create_mlp(vs: &nn::Path, embedding_dimension: i64) -> impl nn::Module {

    let d: i64 = embedding_dimension;
    let q: i64 = embedding_dimension * 3;

    let projection_1dq = vs.var("projection_1dq", &[1, d, q], generate_init());
    let expansion_1qd = vs.var("expansion_1qd", &[1, q, d], generate_init());
    let layer_norm = create_layer_norm(vs, embedding_dimension);

    

    nn::func(move |input_bcd: &tch::Tensor| {

        let x_bcd = layer_norm.forward(input_bcd);
        let x_bcq = &x_bcd.matmul(&projection_1dq);
        let activations_bcq = x_bcq.gelu("none");
        activations_bcq.matmul(&expansion_1qd) + input_bcd // https://arxiv.org/abs/1512.03385
    })
}
