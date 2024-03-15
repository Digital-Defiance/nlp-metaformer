

use tch::{Device, Kind, TchError, Tensor};


extern "C" {
    fn add_tensors_cuda(
        result: *mut torch_sys::C_tensor,
        a: *const torch_sys::C_tensor,
        b: *const torch_sys::C_tensor
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
                a.as_ptr(),
                b.as_ptr()
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
