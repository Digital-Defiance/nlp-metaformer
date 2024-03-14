



__global__ void add_constant_kernel(float *x, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] += n;
}

extern "C" void add_constant_cuda(float *x, int n) {
    add_constant_kernel<<<(n + 255) / 256, 256>>>(x, n);
}
