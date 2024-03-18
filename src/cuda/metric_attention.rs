

use libc::c_float;
use tch::{Device, Kind, TchError, Tensor};
use torch_sys::C_tensor;

type TensorPTR = *mut C_tensor;
type ImmutableTensorPTR = *const C_tensor;

extern "C" {
    fn f_metric_tensor_attention(out: *mut TensorPTR, input_bcd: ImmutableTensorPTR,  metric_1nkk: TensorPTR);
}

pub trait MetricAttention {
    fn metric_tensor_attention(&self, metric_1nkk: &mut Tensor) -> Tensor;
}

impl MetricAttention for Tensor {
    fn metric_tensor_attention(&self, metric_1nkk: &mut Tensor) -> Tensor {
        let mut c_tensors = [std::ptr::null_mut(); 1];
        unsafe {
            f_metric_tensor_attention(
                c_tensors.as_mut_ptr(),
                self.as_ptr(),
                metric_1nkk.as_mut_ptr()
            );
            Tensor::from_ptr(c_tensors[0])
        }

    }
}


#[test]
fn test_add_constant(){

    let device = Device::cuda_if_available();

    let data: &[c_float; 2] = &[5., 25.];
    let input_bcd = Tensor::from_slice(data).to(device);
    let mut metric_1nkk = Tensor::from_slice(data).to(device);
    let output_bcd = input_bcd.metric_tensor_attention( &mut metric_1nkk);
    output_bcd.print();

   
   
   
   
   
   
   //  assert!(result.equal(&c));



}
