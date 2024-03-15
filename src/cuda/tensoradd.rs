

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
    fn add_tensors(self, a: &Tensor, b: &Tensor) -> Tensor;
}

impl AddTensors for Tensor {
    fn add_tensors(mut self, a: &Tensor, b: &Tensor) -> Tensor {
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
    let a = Tensor::from_slice(&[5., 25.]).to(device);
    let b = Tensor::from_slice(&[1., 2.]).to(device);
    let _c = Tensor::from_slice(&[6., 27.]).to(device);

    let result = Tensor::zeros(&[1, 2], (Kind::Float, device));
    let result = result.add_tensors(&a, &b);

    
    result.print();
   //  assert!(result.equal(&c));



}
