

use tch::{Device, Kind, TchError, Tensor};

use torch_sys::C_tensor;

extern "C" {
    fn add_constant_cpp(a: *const f64, b: *const f64) -> *mut f64;
}

pub trait AddConstantRust {
    fn add_constant_rust(a: *const f64, b: *const f64) -> *mut f64;
}

impl AddConstantRust for Tensor {
    fn add_constant_rust(a: *const f64, b: *const f64) -> *mut f64 {
        unsafe {
            add_constant_cpp(a, b)
        }
        
        
        /* 
         unsafe {
            let c_tensor = self.as_mut_ptr();
            add_constant_cpp(c_tensor, c);
            // Ok(Tensor::from_ptr(c_tensor))
         }
        
        // let mut c_tensors = [std::ptr::null_mut(); 1];
        // unsafe_torch_err!(atg_abs(c_tensors.as_mut_ptr(), self.c_tensor));

 
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

    // et x = Tensor::ones(&[3], (Kind::Int, Device::Cpu));

    let a = vec![1., 2.];
    let b = vec![1., 2.];

    let _x = Tensor::add_constant_rust(
        a.as_ptr(),
        b.as_ptr()
    );

    

    // assert_eq!(x, 3);
    println!("test");

}