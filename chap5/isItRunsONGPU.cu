// Author: Elin.Liu
// Date: 2022-11-26 20:43:16
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-11-26 20:43:16

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel(void)
{
    while (1)
    {
        printf("Hello World!\n");
    }
}

int main(void)
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}