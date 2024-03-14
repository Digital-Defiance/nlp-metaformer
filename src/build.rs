

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
"; */







fn main() {

    let files = vec![
        "src/attention/metric.cpp",
        "src/attention/metric_kernel.cu"
    ];
    cc::Build::new()
    .cuda(true)
    .files(files)
    
    // .file()
    .flag("-std=c++17")
    .warnings(false)
    .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", 0))
    .include("/opt/conda/lib/python3.10/site-packages/torch/include")
    .include("/opt/conda/lib/python3.10/site-packages/torch")
    .include("/opt/conda/lib/python3.10/site-packages/torch/include/torch/csrc/api/include")
    .include("/opt/conda/include/python3.10")
    // .flag(&format!("-Wl,-rpath=/opt/conda/lib/python3.10/site-packages/torch/lib"))
    .compile("metric.a");
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