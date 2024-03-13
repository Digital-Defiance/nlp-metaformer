
use tch::nn;
use tch::Tensor;

use crate::metaformer::commons::generate_init;

#[derive(Debug)]
pub struct QuadraticAttention {
    projections_1ndq: Tensor,
    metric_tensors_1nqq: Tensor,
    adapter_1pd: Tensor,
    sqrt_q: f64,
    cp: (i64, i64),
}


/// Performs self attention N times using the quadratic form $xW_nx.T$ where $W_n$ is a learnable matrix.
/// This is an early version of the metric self attention, where $W$ is forced to have the properties a metric tensor.
/// https://arxiv.org/abs/2111.11418 - evidence that any of the attention mechanisms might have similar performance 
impl QuadraticAttention {
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
    
        let projections_1ndq = vs_path.var("projections_1ndq", &[1, n, d, q], generate_init());
        let metric_tensors_1nqq = vs_path.var("metric_tensors_1nqq", &[1, n, q, q], generate_init());
        let adapter_1pd = vs_path.var("adapter_1pd", &[1, p, d], generate_init());
    
        let sqrt_q = f64::sqrt(q as f64);
        QuadraticAttention { 
            projections_1ndq, 
            metric_tensors_1nqq,
            adapter_1pd,
            sqrt_q,
            cp: (c, p)
        }
    }
}

// Implement the nn::Module trait for QuadraticAttention.
impl nn::Module for QuadraticAttention {
    fn forward(&self, x_bcd: &Tensor) -> Tensor {

        let b = x_bcd.size()[0];
        // assert_eq!(x_bcd.size(), vec![b, c, d]);

        // Apply n projections to the input 
        let x_b1cd = &x_bcd.unsqueeze(1);
        let x_bncq = &x_b1cd.matmul(&self.projections_1ndq);
        // debug_assert_eq!(x_bncq.size(), vec![b, n, c, q]);

        // Use n custom dot products to generate n score tables
        let x_bnqc = &x_bncq.transpose(-1, -2);
        let dotproducts_bncc = &x_bncq.matmul(&self.metric_tensors_1nqq.matmul(x_bnqc));
        // debug_assert!(dotproducts_bncc.size() == vec![b, n, c, c]);
    
        // From scaled dot product attention introduced in https://arxiv.org/abs/1706.03762
        let scaled_dotproducts_bncc = &dotproducts_bncc.divide_scalar(self.sqrt_q);

        let softmaxed_scaled_dotproducts_bncc = &scaled_dotproducts_bncc.softmax(-1, tch::kind::Kind::Float);
        let y_bnqc = &x_bncq.transpose(-1, -2).matmul(softmaxed_scaled_dotproducts_bncc);
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


