

use tch::{Device, Kind, TchError, Tensor};

use torch_sys::C_tensor;

extern "C" {
    fn add_tensors_cpp(
        result: *mut torch_sys::C_tensor,
        a: *const torch_sys::C_tensor,
        b: *const torch_sys::C_tensor
    );
}

pub trait AddTensors {
    fn add_tensors(self,a: &Tensor, b: &Tensor) -> Tensor;
}

impl AddTensors for Tensor {
    fn add_tensors(mut self, a: &Tensor, b: &Tensor) -> Tensor {
        unsafe {
            add_tensors_cpp(
                self.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr()
            );
        }
        self
    }
}


#[test]
fn test_add_constant(){


    let a = Tensor::from_slice(&[5., 25.]);
    let b = Tensor::from_slice(&[1., 2.]);
    let _c = Tensor::from_slice(&[6., 27.]);

    let result = Tensor::zeros(&[1, 2], (Kind::Float, Device::Cpu));
    let result = result.add_tensors(&a, &b);

    
    result.print();
   //  assert!(result.equal(&c));



}
