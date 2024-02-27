
use tch::nn::{ self, Module };

use crate::metaformer::commons;
use commons::generate_init;


fn create_config() -> nn::EmbeddingConfig{
    nn::EmbeddingConfig{
    
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
    }
}



/// Creates the embedder module, which transforms integer valued tokens into positionally encoded embeddings.
pub fn create_embedder_module(
    vs: &nn::Path,
    embedding_dimension: i64,
    size_of_vocabolary: i64,
    size_of_context_window: i64,
) -> impl nn::Module {
 
    let config = create_config();
    let d: i64 = embedding_dimension;
    let v: i64 = size_of_vocabolary;
    let c: i64 = size_of_context_window;

    let vocabolary_vd = nn::embedding(vs, v, d, config);
    let positional_encoding_cd = nn::embedding(vs, c, d, config);

    nn::func(move |x_bc: &tch::Tensor| {

        let indexes = tch::Tensor::arange(x_bc.size()[1], (tch::Kind::Int, tch::Device::Cpu));
        vocabolary_vd.forward(&x_bc) + positional_encoding_cd.forward(&indexes)
    })
}







#[cfg(test)]
mod tests {
    use super::*; 
    use tch::{nn, Device, Kind, Tensor};
    use tch::nn::Module;


    #[test]
    pub fn test_layer(){


        let vs = nn::VarStore::new(Device::Cpu);
        let vs_path = &vs.root();
    
        let b = 10;
        let c = 5;
        let d = 4;
        let n = 2;
        let q = 2;

        let input_bc = Tensor::randint( 50, &[b, c],  (Kind::Int, Device::Cpu));
        let layer = create_embedder_module(vs_path, d, 50, c);
        let output_bcd = layer.forward(&input_bc);

        debug_assert!(output_bcd.size() == vec![b, c, d]);

    }

}