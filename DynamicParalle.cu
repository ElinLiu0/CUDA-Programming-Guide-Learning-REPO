// Author: Elin.Liu
// Date: 2022-12-16 18:26:04
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-12-16 18:26:04

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernelB(void)
{
    printf("Hello CUDA!\n");
}

__global__ void kernelA(void)
{
    /* When calling kernel function in dynamic paralism
    it need to add a compile para which -rdc=true
    */
    kernelB<<<1, 1>>>();
}

__global__ void calit(int x, int y)
{
    // Since the takeit function can directly takes the a and b
    // from host buffer,then calit can also directly take it
    // from GPU buffer
    printf("x + y = %d\n", x + y);
}

__global__ void takeit(int a, int b)
{
    calit<<<1, 1>>>(a, b);
}
int main(void)
{
    kernelA<<<1, 1>>>();
    takeit<<<1, 1>>>(1, 1);
    cudaDeviceSynchronize();
    return 0;
}