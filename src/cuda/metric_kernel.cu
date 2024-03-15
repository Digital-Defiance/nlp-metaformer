

__global__ void add_vectors_kernel(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] += a[i] + b[i];
}


extern "C" {
    void add_vectors_cuda(float *a, float *b, float *c) {
        int numBlocks = 1;
        int numThreadsPerBlock = 2;
        add_vectors_kernel<<<numBlocks, numThreadsPerBlock>>>(a, b, c);
    }
}
