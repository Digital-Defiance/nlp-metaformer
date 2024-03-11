
use self::mlp::create_mlp;
use self::embedder::create_embedder_module;
use crate::attention::quadratic_form::QuadraticAttention;
use crate::attention::scaled_dot_product::ScaledDotProductAttention;

pub mod layer_norm;
pub mod commons;
pub mod embedder;
pub mod mlp;


use layer_norm::create_layer_norm;
use commons::generate_init;
use tch::nn;
use tch::nn::Module;
use crate::attention::avg_pooling::AvgPooling;
use tch::nn::func;
use tch::Tensor;
use tch;



/// Defines structure of the metaformer model
/// GPT2 paper - https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
/// MetaFormer paper - TODO
#[derive(Debug)]
pub struct MetaFormer {
    embedding_dimension: i64,
    size_of_context_window: i64,
    layers: Vec<Box<dyn Module>>,
}

impl Module for MetaFormer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.layers[0].forward(xs);
        self.layers.iter().skip(1).fold(xs, |xs, layer| layer.forward(&xs))
    }
}


/// Creates a new empty metaformer layer.
pub fn metaformer(
    vs_path: &nn::Path,
    embedding_dimension: i64,
    size_of_vocabolary: i64,
    size_of_context_window: i64,
    device: tch::Device,
) -> MetaFormer {

    let embedder = create_embedder_module(
        vs_path, 
        embedding_dimension,
        size_of_vocabolary,
        size_of_context_window,
        device
    );

    MetaFormer {
        embedding_dimension,
        size_of_context_window,
        layers: vec![ Box::new(embedder) ]
    }
}



impl MetaFormer {

    pub fn add_mlp(self, vs_path: &nn::Path) -> Self {
        let layer = create_mlp(vs_path, self.embedding_dimension);
        self.add(vs_path, layer)
    }


    pub fn add_avg_pooling(self, vs_path: &nn::Path, kernel_size: i64) -> Self {
        let layer = AvgPooling::new(kernel_size);
        self.add(vs_path, layer)
    }

    pub fn add_scaled_dot_product(self, vs_path: &nn::Path, number_of_heads: i64) -> Self {
        let layer = ScaledDotProductAttention::new(
            vs_path,
            number_of_heads,
            self.embedding_dimension,
            self.embedding_dimension / number_of_heads,
            self.size_of_context_window,
        );
        self.add(vs_path, layer)
    }

    pub fn add_quadratic_form(self, vs_path: &nn::Path, number_of_heads: i64) -> Self {
        let layer = QuadraticAttention::new(
            vs_path,
            number_of_heads,
            self.embedding_dimension,
            self.embedding_dimension / number_of_heads,
            self.size_of_context_window,
        );
        self.add(vs_path, layer)
    }


    pub fn finish(mut self, vs_path: &nn::Path, output_tokens: i64) -> Self {
        
        let d = self.embedding_dimension;
        let t = output_tokens;

        let linear_norm = create_layer_norm(vs_path, self.embedding_dimension);
        let projection_1dt = vs_path.var("projection_1dt", &[1, d, t], generate_init());

        let final_layer = nn::func(move |x_bcd| {
            let y_bcd = &linear_norm.forward(x_bcd);
            y_bcd.matmul(&projection_1dt)
        });

        self.layers.push(Box::new(final_layer));
        self
    }


    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    fn add<M: Module + 'static>(mut self, vs_path: &nn::Path, layer: M) -> Self {

        let layer_norm = create_layer_norm(vs_path, self.embedding_dimension);

        self.layers.push(Box::new(func(
            move |x: &tch::Tensor| layer.forward(&layer_norm.forward(x)) + x
        )));
        self
    }
}




















































/*
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

    kernel_size: Option<i64>,

    
}

*/




/*
#[test]
pub fn test_model_creation(){

    let metaformer: MetaFormer = MetaFormer::new(
        32,
        3,
        2,
        10,
        5,
        5,
    );

    let vs = nn::VarStore::new(Device::Cpu);
    let vs_path = &vs.root();
    let _quadratic_network = metaformer.create(vs_path, AttentionKind::Quadratic, Device::Cpu);
    // let _transformer_network = metaformer.create(vs_path, AttentionKind::ScaledDotProduct);
    // let _metric_network = metaformer.create(vs_path, AttentionKind::Metric);
}
 */