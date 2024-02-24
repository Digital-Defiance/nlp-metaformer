use self::commons::MetaformerParameters;
use self::mlp::create_mlp;
use self::embedder::create_embedder_module;

pub mod layer_norm;
pub mod commons;
pub mod embedder;
pub mod mlp;

use tch::nn;

impl MetaformerParameters {


    pub fn create(&self, vs: &nn::Path) -> impl nn::Module 
    {

        let mut model = nn::seq();
        model = model.add(create_embedder_module(vs, self));
        
        for _ in 0..hp.model_depth  {
            // model = model.add(create_self_attention(vs, hp));
            model = model.add(create_mlp(vs, self.embedding_dimenson));
        }

        // model = model.add(create_output_tokenizer(vs_path, hyper_parameters.output_tokens))
        model
    }
}