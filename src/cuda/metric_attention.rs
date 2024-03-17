

use libc::c_float;
use tch::{Device, Kind, TchError, Tensor};
use torch_sys::C_tensor;

type TensorPTR = *mut C_tensor;
type ImmutableTensorPTR = *const C_tensor;

extern "C" {
    fn f_metric_tensor_attention(input_bcd: ImmutableTensorPTR,  output_bcd: TensorPTR,  metric_1nkk: TensorPTR);
}

pub trait MetricAttention {
    fn metric_tensor_attention(&self, output_bcd: &mut Tensor, metric_1nkk: &mut Tensor) ;
}

impl MetricAttention for Tensor {
    fn metric_tensor_attention(&self, output_bcd: &mut Tensor, metric_1nkk: &mut Tensor) {
        unsafe {
            f_metric_tensor_attention(
                self.as_ptr(),
                output_bcd.as_mut_ptr(),
                metric_1nkk.as_mut_ptr()
            );
        }
    }
}


#[test]
fn test_add_constant(){

    let device = Device::cuda_if_available();

    let data: &[c_float; 2] = &[5., 25.];
    let input_bcd = Tensor::from_slice(data).to(device);
    let mut metric_1nkk = Tensor::from_slice(data).to(device);

    let mut output_bcd = Tensor::zeros(&[1, 2], (Kind::Float, device));
    input_bcd.metric_tensor_attention(&mut output_bcd, &mut metric_1nkk);
    output_bcd.print();
   //  assert!(result.equal(&c));



}
