
use tch::nn;
use tch::Tensor;



#[derive(Debug)]
pub struct Identity { }

impl Identity {
    pub fn new() -> Self {
        Identity { }
    }
}

impl nn::Module for Identity {
    fn forward(&self, x_bcd: &Tensor) -> Tensor {   
        x_bcd.g_mul_scalar(1)
    }
}

