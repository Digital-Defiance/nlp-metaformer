#include <cuda_runtime.h> 







template <typename scalar_t>
__global__ void add_tensors_kernel(
    scalar_t* a,
    scalar_t* b,
    scalar_t* c
) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    c[idx] += a[idx] + b[idx];

}



const size_t SIZE_OF_VEC2 = 2*sizeof(float);

extern "C" {

    void add_tensors_cuda(float *a, float *b, float *c) {
        int numBlocks = 1;
        int numThreadsPerBlock = 2;


        // Allocate memory for arrays d_A, d_B, and d_C on device
        float *d_A, *d_B, *d_C;
    
        cudaMalloc(&d_A, SIZE_OF_VEC2);
        cudaMalloc(&d_B, SIZE_OF_VEC2);
        cudaMalloc(&d_C, SIZE_OF_VEC2);

        // Copy data from host arrays A and B to device arrays d_A and d_B
        cudaMemcpy(d_A, a, SIZE_OF_VEC2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, b, SIZE_OF_VEC2, cudaMemcpyHostToDevice);


        add_tensors_kernel<float><<<numBlocks, numThreadsPerBlock>>>(d_A, d_B, d_C);
    
    	cudaMemcpy(c, d_C, SIZE_OF_VEC2, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}
