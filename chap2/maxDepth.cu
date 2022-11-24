// Author: Elin.Liu
// Date: 2022-11-24 15:02:12
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-11-24 15:02:12

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cuda.h"

int main(void)
{
    // 获取第一个GPU的名字
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // 获取GPU的最大栈空间
    size_t maxStackSize;
    cudaDeviceGetLimit(&maxStackSize, cudaLimitStackSize);
    printf("GPU name: %s max stack size: %d bytes\n", prop.name, maxStackSize);
    // 调用cudaDeviceSetLimit设置GPU的最大栈空间
    size_t wanted_stack_memory = 2048;
    cudaDeviceSetLimit(cudaLimitStackSize, wanted_stack_memory);
    // 再次获取GPU的最大栈空间
    cudaDeviceGetLimit(&maxStackSize, cudaLimitStackSize);
    printf("After setting though,GPU stack memory now is %d bytes.\n", maxStackSize);
}