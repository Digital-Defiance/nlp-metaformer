// use libc::c_float;
// use tch::Device;
use tch::Tensor;
// use tch::{
//     kind,
//     // nn::{self, OptimizerConfig},
//     Device,
//     Kind,
//     TchError,
//     Tensor,
// };
use torch_sys::C_tensor;
// use crate::{metaformer::commons::generate_init, optimizer::build_optimizer};

type TensorPTR = *mut C_tensor;
type ImmutableTensorPTR = *const C_tensor;

extern "C" {
    fn f_metric_tensor_attention(
        out: *mut TensorPTR,
        input_bcd: ImmutableTensorPTR,
        metric_1nkk: TensorPTR,
    );
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
                metric_1nkk.as_mut_ptr(),
            );
            Tensor::from_ptr(c_tensors[0])
        }
    }
}

#[test]
fn test_backwards_pass() {
    println!("Running backwards pass...");

    let device = Device::cuda_if_available();

    let data = &[5., 25.];

    let input_bcd = Tensor::zeros(&[2, 2], (kind::Kind::Float, device));

    let gt_output_bcd = Tensor::zeros(&[2, 2], (kind::Kind::Float, device));

    //  assert!(result.equal(&c));
    let vs: nn::VarStore = nn::VarStore::new(device);
    let mut opt = match tch::nn::Adam::default().build(&vs, 0.8e-1) {
        Ok(result) => result,
        Err(err) => panic!("Error while building optimizer: {}", err),
    };

    let mut metric_1nkk = vs.root().var("metric_1nkk", &[1, 2], generate_init());

    // Tensor::from_slice(data).to(device).set_requires_grad(true);
    for _ in 0..20 {
        let output_bcd = input_bcd.metric_tensor_attention(&mut metric_1nkk);
        let loss = output_bcd.mse_loss(&gt_output_bcd, tch::Reduction::Mean);
        opt.backward_step(&loss);
        loss.print();
        metric_1nkk.print();
    }
}
