

use libc::c_float;
use tch::{Device, Kind, TchError, Tensor};
use torch_sys::C_tensor;

type TensorPTR = *mut C_tensor;

extern "C" {
    fn add_tensors_cuda(
        result: TensorPTR,
        a: TensorPTR,
        b: TensorPTR
    );
}

pub trait AddTensors {
    fn add_tensors(self, a: &mut Tensor, b: &mut Tensor) -> Tensor;
}

impl AddTensors for Tensor {
    fn add_tensors(mut self, a: &mut Tensor, b: &mut Tensor) -> Tensor {
        unsafe {
            add_tensors_cuda(
                self.as_mut_ptr(),
                a.as_mut_ptr(),
                b.as_mut_ptr()
            );
        }
        self
    }
}


#[test]
fn test_add_constant(){

    let device = Device::cuda_if_available();

    let data: &[c_float; 2] = &[5., 25.];
    let mut a = Tensor::from_slice(data).to(device);
    let mut b = Tensor::from_slice(data).to(device);
    Tensor::matmul
    let result = Tensor::zeros(&[1, 2], (Kind::Float, device));
    let result = result.add_tensors(&mut a, &mut b);

    
    result.print();
   //  assert!(result.equal(&c));



}
