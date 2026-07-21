#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        float max_val = input[0];
        for(int i = 1; i < N; i++)
            if(input[i] > max_val) max_val = input[i];
        for(int i = 0; i < N; i++) sum += expf(input[i] - max_val);
        output[idx] = expf(input[idx] - max_val) / sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}