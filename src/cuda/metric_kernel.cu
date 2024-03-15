



/// @brief 
/// @param a 
/// @param b 
/// @return 
__global__ void add_constant_kernel(float *a, float *b) {
    /* int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] += n; */
    
}


extern "C" {

    /// @brief 
    /// @param a 
    /// @param b 
    /// @return 
    float* add_constant_cuda(float *a, float *b) {
        add_constant_kernel<<<2,2>>>(a, b);
        return a;
    }
}
