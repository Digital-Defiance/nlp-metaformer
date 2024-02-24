use self::commons::MetaformerParameters;
use self::mlp::create_mlp;

pub mod layer_norm;
pub mod commons;
pub mod embedder;
pub mod mlp;


pub fn create_metaformer(vs: &nn::Path, hp: &ModelParameters) -> impl nn::Module 
{

    let mut model = nn::seq();
    model = model.add(create_embedder_module(vs, hp));
    
    for _ in 0..hp.model_depth  {
        model = model.add(create_self_attention(vs, hp));
        model = model.add(create_mlp(vs, hp.embedding_dimenson));
    }

    // model = model.add(create_output_tokenizer(vs_path, hyper_parameters.output_tokens))
    model
}
