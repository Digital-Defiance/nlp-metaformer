

use tch::{Device, Kind, TchError, Tensor};

use torch_sys::C_tensor;

extern "C" {
    fn add_vectors_cpp(a: *const f64, b: *const f64, n: i64) -> *mut f64;
}

pub trait AddConstantRust {
    fn add_vectors_rust(a: *const f64, b: *const f64, n: i64) -> *mut f64;
}

impl AddConstantRust for Tensor {
    fn add_vectors_rust(a: *const f64, b: *const f64, n: i64) -> *mut f64 {
        unsafe {
            add_vectors_cpp(a, b, n)
        }
    }
}


#[test]
fn test_add_constant(){

    // et x = Tensor::ones(&[3], (Kind::Int, Device::Cpu));

    let a = vec![1., 2.];
    let b = vec![1., 2.];

    let x = Tensor::add_vectors_rust(
        a.as_ptr(),
        b.as_ptr(),
        2
    );

    unsafe {


        let y = *x;
        let z: f64 = *(x.wrapping_add(1));

        assert_eq!(y, 2.);
        assert_eq!(z, 4.);
    }

    // assert_eq!(x, 3);
    println!("test");

}
