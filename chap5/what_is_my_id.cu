// Author: Elin.Liu
// Date: 2022-11-26 13:53:29
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-11-26 13:53:29

#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_runtime.h"

// 定义ID查询函数
__global__ void what_is_my_id(
    unsigned int *const block,
    unsigned int *const thread,
    unsigned int *const warp,
    unsigned int *const calc_thread)
{
    /*线程ID是线程块的索引 x 线程块的大小 + 线程数量的起始点*/
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;

    /*线程束 = 线程ID / 内置变量warpSize*/
    warp[thread_idx] = thread_idx / warpSize;

    calc_thread[thread_idx] = thread_idx;
}

// 定义数组大小
#define ARRAY_SIZE 1024
// 定义数组字节大小
#define ARRAY_BYTES ARRAY_SIZE * sizeof(unsigned int)

// 声明主机下参数
unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

// 定义主函数
int main(void)
{
    // 总线程数量为 2 x 64 = 128
    // 初始化线程块和线程数量
    const unsigned int num_blocks = 2;
    const unsigned int num_threads = 64;
    char ch;

    // 声明设备下参数
    unsigned int *gpu_block, *gpu_thread, *gpu_warp, *gpu_calc_thread;

    // 声明循环数量
    unsigned int i;

    // 为设备下参数分配内存
    cudaMalloc((void **)&gpu_block, ARRAY_BYTES);
    cudaMalloc((void **)&gpu_thread, ARRAY_BYTES);
    cudaMalloc((void **)&gpu_warp, ARRAY_BYTES);
    cudaMalloc((void **)&gpu_calc_thread, ARRAY_BYTES);

    // 调用核函数
    what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);

    // 将设备下参数复制到主机下
    cudaMemcpy(cpu_block, gpu_block, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread, gpu_thread, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_warp, gpu_warp, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // 释放GPU内存
    cudaFree(gpu_block);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);

    // 循环打印结果
    for (i = 0; i < ARRAY_SIZE; i++)
    {
        printf("Calculated Thread: %d - Block: %d - Warp: %d - Thread: %d\n", cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
    }
    return 0;
}