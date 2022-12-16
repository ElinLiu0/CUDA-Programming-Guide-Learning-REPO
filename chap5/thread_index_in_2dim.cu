// Author: Elin.Liu
// Date: 2022-11-27 14:32:53
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-11-27 14:32:53

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <cuda_runtime.h>

__global__ void kernel(void)
{
    // Get CUDA Thread Index on X dimension
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Get CUDA Thread Index on Y dimension
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("Current Thread Position at x : %d, y : %d\n", x, y);
}

int main(void)
{
    // Define CUDA Grid Size
    dim3 grid(2, 2);
    // Define CUDA Block Size
    dim3 block(2, 2);
    // Launch CUDA Kernel
    kernel<<<grid, block>>>();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    // Check for any errors launching the kernel
    cudaGetLastError();
    return 0;
}