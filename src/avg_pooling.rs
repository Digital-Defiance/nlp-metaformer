
use tch::nn::Module;
use tch::Tensor;



// implicit zero paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
// const DEFAULT_PADDING: i64 = 0;

/// when True, will use ceil instead of floor in the formula to compute the output shape. Default: False
const DEFAULT_CEIL_MODE: bool = false;

/// when True, will include the zero-padding in the averaging calculation. Default: True
const DEFAULT_COUNT_INCLUDE_PAD: bool = false;

/// if specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: None
const DEFAULT_DIVISOR_OVERRIDE: core::option::Option<i64> = None;


/// Average pooling layer
#[derive(Debug)]
pub struct AvgPooling {
    /// Size of the pooling region. Can be a single number or a tuple (kH, kW)
    kernel_size: i64,
}


impl AvgPooling {
    pub fn new(kernel_size: i64) -> Self {
        AvgPooling {  kernel_size  }
    }
}

impl Module for AvgPooling {
    fn forward(&self, x_bcd: &Tensor) -> Tensor {
        x_bcd.avg_pool2d(
            self.kernel_size,
            // stride of the pooling operation. Can be a single number or a tuple (sH, sW). Default: kernel_size
            1,
            self.kernel_size / 2,
            DEFAULT_CEIL_MODE, 
            DEFAULT_COUNT_INCLUDE_PAD, 
            DEFAULT_DIVISOR_OVERRIDE, 
        ) - x_bcd
    }
}


#[cfg(test)]
mod tests {
    use super::*; 
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_avg_pooling_forward() {
       let kernel_size = 5;
       let pooling = AvgPooling::new(kernel_size);

  
       let b = 32;
       let c = 300;
       let d = 64;

       let x_bcd = Tensor::ones(&[b, c, d], (Kind::Float, Device::Cpu));
       let res = pooling.forward(&x_bcd);
        print!("here");
        

        assert_eq!(res.size()[0], x_bcd.size()[0]);
        assert_eq!(res.size()[1], x_bcd.size()[1]);
        assert_eq!(res.size()[2], x_bcd.size()[2]);



       



    }
}