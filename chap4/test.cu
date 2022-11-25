// Author: Elin.Liu
// Date: 2022-11-25 11:16:22
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-11-25 11:16:22

#include "cuda_runtime.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>

/*验证CUDA CALL宏是否可用*/
int main(void)
{
    cudaError_t cudaStatus;
    cudaDeviceProp prop;
    cudaStatus = cudaGetDeviceProperties(&prop, 0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Get CUDA Device Properties failed!\n");
        return 1;
    }
    else
    {
        printf("GPU Name : %s\n", prop.name);
        return 0;
    }
}
