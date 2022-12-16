// Author: Elin.Liu
// Date: 2022-11-25 21:09:22
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-11-25 21:09:22

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

/*cpu 中的矩阵乘法*/
__host__ void some_func(int *a, int *b, int *c)
{
    int i;

    for (i = 0; i < 128; i++)
    {
        a[i] = b[i] * c[i];
    }
}

/*gpu 中的矩阵乘法*/
__global__ void some_kernel_func(int *a, int *b, int *c)
{
    // 初始化线程ID
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    // 对数组元素进行乘法运算
    a[i] = b[i] * c[i];
    // 打印打前处理的进程ID
    // 可以看到blockIdx并非是按照顺序启动的，这也说明线程块启动的随机性
    printf("blockIdx.x = %d,blockDimx.x = %d,threadIdx.x = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
}

int main(void)
{
    // 初始化指针元素
    int *a, *b, *c;
    // 初始化GPU指针元素
    int *gpu_a, *gpu_b, *gpu_c;
    // 初始化数组大小
    int size = 128 * sizeof(int);
    // 为CPU指针元素分配内存
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);
    // 为GPU指针元素分配内存
    cudaMalloc((void **)&gpu_a, size);
    cudaMalloc((void **)&gpu_b, size);
    cudaMalloc((void **)&gpu_c, size);
    // 初始化数组元素
    for (int i = 0; i < 128; i++)
    {
        b[i] = i;
        c[i] = i;
    }
    // 将数组元素复制到GPU中
    cudaMemcpy(gpu_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c, c, size, cudaMemcpyHostToDevice);
    // 执行GPU核函数
    some_kernel_func<<<4, 32>>>(gpu_a, gpu_b, gpu_c);
    // 将GPU中的结果复制到CPU中
    cudaMemcpy(a, gpu_a, size, cudaMemcpyDeviceToHost);
    // 释放GPU和CPU中的内存
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    free(a);
    free(b);
    free(c);
    return 0;
}