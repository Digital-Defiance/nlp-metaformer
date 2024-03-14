
/*
__global__ void add_constant_kernel(int a, int b) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] += n;
}
*/

extern "C" {
    float *add_constant_cuda(float *a, float *b) {
    return a;

    // add_constant_kernel<<<(n + 255) / 256, 256>>>(x, n);
}
}
