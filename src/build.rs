

use cc;



/*
sadas
const TORCH_VERSION: &str = "2.2.0";
const PYTHON_PRINT_PYTORCH_DETAILS: &str = r"
import torch
from torch.utils import cpp_extension
print('LIBTORCH_VERSION:', torch.__version__.split('+')[0])
print('LIBTORCH_CXX11:', torch._C._GLIBCXX_USE_CXX11_ABI)
for include_path in cpp_extension.include_paths():
  print('LIBTORCH_INCLUDE:', include_path)
for library_path in cpp_extension.library_paths():
  print('LIBTORCH_LIB:', library_path)
";

const PYTHON_PRINT_INCLUDE_PATH: &str = r"
import sysconfig
print('PYTHON_INCLUDE:', sysconfig.get_path('include'))
"; 


export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
export NVCC=/usr/local/cuda/bin/nvcc
export CARGO_PROFILE_TEST_BUILD_OVERRIDE_DEBUG=true
*/







fn main() {

    let files = vec![
       // "src/cuda/vecadd_kernel.cu",
        // "src/cuda/vecadd.cpp",


        "src/cuda/tensoradd_kernel.cu",
      // "src/cuda/tensoradd.cpp",

    ];

    for file in &files {
      println!("cargo:rerun-if-changed={}", file);
    }
    cc::Build::new()
    .cuda(true)
    .pic(true)
    
    .files(files)
    
    .flag("-std=c++17")
    .warnings(false)
    .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", 0))
   // .flag("-c")
    .include("/opt/conda/lib/python3.10/site-packages/torch/include")
    .include("/opt/conda/lib/python3.10/site-packages/torch")
    .include("/opt/conda/lib/python3.10/site-packages/torch/include/torch/csrc/api/include")
    .include("/opt/conda/include/python3.10")

    // .flag(&format!("-Wl,-rpath=/opt/conda/lib/python3.10/site-packages/torch/lib"))
    .compile("llm_vc");

}

    /* 
cc::Build::new()
.cpp(true)
.pic(true)
.warnings(false)
.includes(&self.libtorch_include_dirs)
.flag(&format!("-Wl,-rpath={}", self.libtorch_lib_dir.display()))
.flag("-std=c++17")
.flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", self.cxx11_abi))
.files(&c_files)
.compile("tch");


*/