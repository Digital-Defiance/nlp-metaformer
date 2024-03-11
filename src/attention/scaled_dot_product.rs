
use tch::nn;
use tch::Tensor;

pub fn generate_init() -> nn::Init {
    nn::Init::Randn { mean: 0., stdev: 1. }
}



#[derive(Debug)]
pub struct ScaledDotProductAttention {
    query_weights_1ndq: Tensor, 
    key_weights_1ndq: Tensor,
    value_weights_1ndq: Tensor,
    adapter_1pd: Tensor,
    sqrt_q: f64,
    cp: (i64, i64),
}


/// Performs self attention N times using the quadratic form $xW_nx.T$ where $W_n$ is a learnable matrix.
/// This is an early version of the metric self attention, where $W$ is forced to have the properties a metric tensor.
/// https://arxiv.org/abs/2111.11418 - evidence that any of the attention mechanisms might have similar performance 
impl ScaledDotProductAttention {
    pub fn new(
        vs_path: &nn::Path,
        number_of_heads: i64,
        embedding_dimension: i64,
        latent_dimension: i64,
        sequence_length: i64,
    ) -> Self {

        let n = number_of_heads;
        let d = embedding_dimension;
        let c = sequence_length;
        let q = latent_dimension;
        let p = latent_dimension*number_of_heads;

        let query_weights_1ndq = vs_path.var("query_weights_1ndq", &[1, n, d, q], generate_init());
        let key_weights_1ndq = vs_path.var("key_weights_1ndq", &[1, n, d, q], generate_init());
        let value_weights_1ndq = vs_path.var("value_weights_1ndq", &[1, n, d, q], generate_init());
        let adapter_1pd = vs_path.var("adapter_1pd", &[1, p, d], generate_init());
    
        let sqrt_q = f64::sqrt(q as f64);
        ScaledDotProductAttention { 
            query_weights_1ndq, 
            key_weights_1ndq,
            value_weights_1ndq,
            adapter_1pd,
            sqrt_q,
            cp: (c, p)
        }
    }
}

// Implement the nn::Module trait for QuadraticAttention.
impl nn::Module for ScaledDotProductAttention {
    fn forward(&self, x_bcd: &Tensor) -> Tensor {

        let b = x_bcd.size()[0];
        // assert_eq!(x_bcd.size(), vec![b, self.c, self.d]);

        // Apply n projections to the input 
        let x_b1cd = &x_bcd.unsqueeze(1);
        
        let queries_bncq = &x_b1cd.matmul(&self.query_weights_1ndq);
        let keys_bncq = &x_b1cd.matmul(&self.key_weights_1ndq);
        let values_bncq = &x_b1cd.matmul(&self.value_weights_1ndq);

        // debug_assert_eq!(queries_bncq.size(), vec![b, n, c, q]);
        // debug_assert_eq!(keys_bncq.size(), vec![b, n, c, q]);
        // debug_assert_eq!(values_bncq.size(), vec![b, n, c, q]);

        // Use n custom dot products to generate n score tables
        let keys_bnqc = &keys_bncq.transpose(-1, -2);
        let scores_bncc = &queries_bncq.matmul(keys_bnqc);
        // debug_assert!(scores_bncc.size() == vec![b, n, c, c]);
    
        // From scaled dot product attention introduced in https://arxiv.org/abs/1706.03762
        let scaled_scores_bncc = &scores_bncc.divide_scalar(self.sqrt_q);

        let softmaxed_scaled_scores_bncc = &scaled_scores_bncc.softmax(-1, tch::kind::Kind::Float);
        let y_bnqc = &values_bncq.transpose(-1, -2).matmul(softmaxed_scaled_scores_bncc);
        // debug_assert!(y_bnqc.size() == vec![b, n, q, c]);

        let y_bcp = &y_bnqc.reshape(&[b, self.cp.0, self.cp.1]);
        // debug_assert!(y_bcp.size() == vec![b, c, p]);
    
        y_bcp.matmul(&self.adapter_1pd)
    }
}




/* 

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

        let input_bcd = Tensor::randn( &[b, c, d],  (Kind::Float, Device::Cpu));
        let layer = quadratic_self_attention_module(vs_path, n, d, q, c);
        let output_bcd = layer.forward(&input_bcd);

        debug_assert!(output_bcd.size() == input_bcd.size());

    }

}

*/
