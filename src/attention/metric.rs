

use tch::{Device, Kind, TchError, Tensor};

// use torch_sys::C_tensor;

use libc::c_float;

type Vec2 = [c_float; 2];

extern "C" {
    fn add_vectors_cpp(a: &Vec2, b: &Vec2, c: &mut Vec2);
}

pub trait AddConstantRust {
    fn add_vectors(a: &Vec2, b: &Vec2, c: &mut Vec2);
}

impl AddConstantRust for Tensor {
    fn add_vectors(a: &Vec2, b: &Vec2, c: &mut Vec2){
        unsafe {
            add_vectors_cpp(a, b, c)
        }
    }
}


#[test]
fn test_add_constant(){

    // et x = Tensor::ones(&[3], (Kind::Int, Device::Cpu));

    println!("test");


    let a: Vec2 = [1., 2.];
    let b: Vec2 = [1., 2.];

    let mut c: Vec2 = [0.0, 0.0];
    Tensor::add_vectors(&a, &b, &mut c);

    assert_eq!(c[0], 2.);

    assert_eq!(c[1], 4.);




}
