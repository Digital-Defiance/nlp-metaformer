

__global__ void add_vectors_kernel(float* result, float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] += a[i] + b[i]; 
}


extern "C" {

    float* add_vectors_cuda(float *a, float *b, int n) {
        float *result;
        add_vectors_kernel(result, a, b, n);
        return result;
    }
}
