use crate::attention::quadratic_form::quadratic_self_attention_module;

use self::mlp::create_mlp;
use self::embedder::create_embedder_module;

pub mod layer_norm;
pub mod commons;
pub mod embedder;
pub mod mlp;

use layer_norm::create_layer_norm;
use commons::generate_init;
use tch::nn;
use tch::nn::Module;

#[derive(PartialEq, Clone, Copy)]





pub enum AttentionKind {
    Quadratic,
    ScaledDotProduct,
    Metric,
}

/// Defines structure of the metaformer model
/// GPT2 paper - https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
/// MetaFormer paper - TODO
pub struct MetaFormer {


    /// Dimension of the vector space that the network
    /// uses internally to represent tokens 
    embedding_dimension: i64,

    /// Number of transformer blocks
    model_depth: i64,

    /// Number of attention modules per transformer block
    number_of_heads: i64,

    /// Maximum number of tokens in the input sequence
    size_of_context_window: i64,

    /// Total number of tokens that the network recognizes
    size_of_vocabolary: i64,

    output_tokens: i64,
}


impl MetaFormer {

    pub fn new(
        embedding_dimension: i64,
        model_depth: i64,
        number_of_heads: i64,
        size_of_context_window: i64,
        size_of_vocabolary: i64,
        output_tokens: i64,
    ) -> MetaFormer {
        MetaFormer {
            embedding_dimension: embedding_dimension,
            model_depth: model_depth,
            number_of_heads: number_of_heads,
            size_of_context_window: size_of_context_window,
            size_of_vocabolary: size_of_vocabolary,
            output_tokens: output_tokens,
        }
    }

    fn create_attention(&self, vs: &nn::Path, kind: AttentionKind) -> impl nn::Module {
        if kind == AttentionKind::Quadratic {
            quadratic_self_attention_module(
                vs,
                self.number_of_heads,
                self.embedding_dimension,
                self.embedding_dimension / self.number_of_heads,
                self.size_of_context_window,
            )
        } else {
            quadratic_self_attention_module(
                vs,
                self.number_of_heads,
                self.embedding_dimension,
                self.embedding_dimension / self.number_of_heads,
                self.size_of_context_window,
            )
        }
    }


    fn create_output_layer(&self, vs: &nn::Path) -> impl nn::Module {

        let d = self.embedding_dimension;
        let t = self.output_tokens;

        let linear_norm = create_layer_norm(vs, self.embedding_dimension);
        let projection_1dt = vs.var("projection_1dt", &[1, d, t], generate_init());

        nn::func(move |x_bcd| {
            let y_bcd = &linear_norm.forward(x_bcd);
            y_bcd.matmul(&projection_1dt)
        })
    }

    pub fn create(&self, vs_path: & nn::Path, kind: AttentionKind,) -> impl nn::Module {

        let mut model = nn::seq().add(
            create_embedder_module(
                vs_path, 
                self.embedding_dimension,
                self.size_of_vocabolary,
                self.size_of_context_window,
        ));

        for _ in 0..self.model_depth  {
            let attention_module = self.create_attention(vs_path, kind);
            model = model.add(attention_module);
            model = model.add(create_mlp(vs_path, self.embedding_dimension));
        }

        model.add(self.create_output_layer(vs_path))
    }
}

