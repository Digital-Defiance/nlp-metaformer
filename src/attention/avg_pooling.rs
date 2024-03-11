
use tch::nn;
use tch::Tensor;

pub fn generate_init() -> nn::Init {
    nn::Init::Randn { mean: 0., stdev: 1. }
}



#[derive(Debug)]
pub struct AvgPooling {

}


impl AvgPooling {
    pub fn new(
        vs_path: &nn::Path,
        number_of_heads: i64,
        embedding_dimension: i64,
        sequence_length: i64,
    ) -> Self {

        AvgPooling { 

        }
    }
}

impl nn::Module for AvgPooling {
    fn forward(&self, x_bcd: &Tensor) -> Tensor {   
        x_bcd
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
