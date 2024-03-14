

use tch::{Device, Kind, TchError, Tensor};

use torch_sys::C_tensor;

extern "C" {
    fn add_constant_cpp(c_tensor: *mut C_tensor, c: i64) -> i64;
}


pub trait AddConstantRust {
    fn add_constant_rust(self, c: i64);
}

impl AddConstantRust for Tensor {
    fn add_constant_rust(mut self, c: i64)  {
        
        
         unsafe {
            let c_tensor = self.as_mut_ptr();
            add_constant_cpp(c_tensor, c);
            // Ok(Tensor::from_ptr(c_tensor))
         }
        
        // let mut c_tensors = [std::ptr::null_mut(); 1];
        // unsafe_torch_err!(atg_abs(c_tensors.as_mut_ptr(), self.c_tensor));

        /* 
        unsafe { // Unsafe because we are calling C++ code
            let result = add_constant_cpp(x, c);
            result
        }

        */
    }
}

/*
pub fn f_abs(&self) -> Result<Tensor, TchError> {
    let mut c_tensors = [std::ptr::null_mut(); 1];
    unsafe_torch_err!(atg_abs(c_tensors.as_mut_ptr(), self.c_tensor));
    Ok(Tensor { c_tensor: c_tensors[0] })
}

 */


#[test]
fn test_add_constant(){

    let x = Tensor::ones(&[3], (Kind::Int, Device::Cpu));
    x.add_constant_rust(2);

    println!("test");

}
