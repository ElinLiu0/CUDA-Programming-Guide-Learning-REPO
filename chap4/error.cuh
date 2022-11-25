// Author: Elin.Liu
// Date: 2022-11-25 11:05:20
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-11-25 11:05:20
#define CUDA_CALL(x)                                     \
    {                                                    \
        const cudaError_t a = (x);                       \
        if (a != cudaSuccess)                            \
        {                                                \
            printf("\n CUDA Error: %s (err_num = %d)\n", \
                   cudaGetErrorString(a), a);            \
            cudaDeviceReset();                           \
            assert(0);                                   \
        }                                                \
    }
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cuda.h"

/*由于几乎所有的CUDA API函数调用，都会产生一个cudaError_t 的错误值，代表是否调用API成功*/

/*如果调用API失败，可以通过cudaGetLastError()函数获取错误值，然后通过cudaGetErrorString()函数获取错误信息*/
/*为了防止重复调用这种错误检查机制，我们可以定义一个宏来实现错误的检查*/

/*在主机函数检查CUDA错误*/
__host__ void cuda_error_check(const char *prefix, const char *postfix)
{
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
        // 重置设备
        cudaDeviceReset();
        wait_exit();
        exit(1);
    }
}