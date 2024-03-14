


extern "C" {
    fn add_constant_cpp(x: i64, c: i64) -> i64;
}

pub fn add_constant_rust(x: i64, c: i64) -> i64 {
    unsafe { // Unsafe because we are calling C++ code
        let result = add_constant_cpp(x, c);
        result
    }
}

#[test]
fn test_add_constant(){
    let res = add_constant_rust(1, 3);
    assert_eq!(res, 4);


    println!("test");

}
