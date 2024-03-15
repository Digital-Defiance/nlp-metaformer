

__global__ void add_vectors_kernel(float* result, float *a, float *b) {
    int i = threadIdx.x;
    result[i] += a[i] + b[i]; 
}


extern "C" {

    void add_vectors_cuda(float *result, float *a, float *b, int n) {
        int numBlocks = 1;
        int threadsPerBlock = n;
        add_vectors_kernel<<<numBlocks, threadsPerBlock>>>(result, a, b);
    }
}
