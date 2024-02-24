
use tch::nn;


pub fn generate_init() -> nn::Init {
    nn::Init::Randn { mean: 0., stdev: 1. }
}

/// Performs self attention N times using the quadratic form $xW_nx.T$ where $W_n$ is a learnable matrix.
/// This is an early version of the metric self attention, where $W$ is forced to have the properties a metric tensor.
/// https://arxiv.org/abs/2111.11418 - evidence that any of the attention mechanisms might have similar performance 
pub fn quadratic_self_attention_module(
    vs_path: &nn::Path,
    n: i64,
    d: i64,
    q: i64,
    c: i64,
) ->  impl nn::Module {

    assert!(d % n == 0, "Embeddings dimension must be divisible by the requested number of heads.");
    debug_assert_eq!(n*q, d);

    let projections_1ndq = vs_path.var("projections_1ndq", &[1, n, d, q], generate_init());
    let metric_tensors_1nqq = vs_path.var("metric_tensors_1nqq", &[1, n, q, q], generate_init());
    let mixer_1dd = vs_path.var("mixer_1dd", &[1, d, d], generate_init());

    debug_assert_eq!(projections_1ndq.size(), vec![1, n, d, q]);
    debug_assert_eq!(metric_tensors_1nqq.size(), vec![1, n, q, q]);
    debug_assert_eq!(mixer_1dd.size(), vec![1, d, d]);

    let sqrt_q = f64::sqrt(q as f64);

 

    nn::func(move |x_bcd| {
    
        let b = x_bcd.size()[0];
        assert_eq!(x_bcd.size(), vec![b, c, d]);


        // Apply n projections to the input 
        let x_b1cd = &x_bcd.unsqueeze(1);
        let x_bncq = &x_b1cd.matmul(&projections_1ndq);
        debug_assert_eq!(x_bncq.size(), vec![b, n, c, q]);


        // Use n custom dot products to generate n score tables
        let x_bnqc = &x_bncq.transpose(-1, -2);
        let dotproducts_bncc = &x_bncq.matmul(&metric_tensors_1nqq.matmul(x_bnqc));
        debug_assert!(dotproducts_bncc.size() == vec![b, n, c, c]);
    
        // From scaled dot product attention introduced in https://arxiv.org/abs/1706.03762
        let scaled_dotproducts_bncc = &dotproducts_bncc.divide_scalar(sqrt_q);

        let softmaxed_scaled_dotproducts_bncc = &scaled_dotproducts_bncc.softmax(-1, tch::kind::Kind::Float);
        let y_bnqc = &x_bncq.transpose(-1, -2).matmul(softmaxed_scaled_dotproducts_bncc);
        debug_assert!(y_bnqc.size() == vec![b, n, q, c]);

        let y_bcd = &y_bnqc.reshape(x_bcd.size());
        debug_assert!(y_bcd.size() == vec![b, c, d]);
    
        y_bcd.matmul(&mixer_1dd)
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

        let input_bcd = Tensor::randn( &[b, c, d],  (Kind::Float, Device::Cpu));
        let layer = quadratic_self_attention_module(vs_path, n, d, q, c);
        let output_bcd = layer.forward(&input_bcd);

        debug_assert!(output_bcd.size() == input_bcd.size());

    }

}