
use tch::nn;
use tch::Tensor;


#[derive(Debug)]
pub struct AvgPooling {

    /// Size of the pooling region. Can be a single number or a tuple (kH, kW)
    kernel_size: i64,

    /// stride of the pooling operation. Can be a single number or a tuple (sH, sW). Default: kernel_size
    stride: i64,

    /// implicit zero paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
    padding: i64,

    /// when True, will use ceil instead of floor in the formula to compute the output shape. Default: False
    ceil_mode: bool,

    /// when True, will include the zero-padding in the averaging calculation. Default: True
    count_include_pad: bool,

    /// if specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: None
    divisor_override: i64,

    number_of_heads: i64,

    embedding_dimension: i64,
    
    sequence_length: i64
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
        x_bcd.avg_pool2d(
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override
        )
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
