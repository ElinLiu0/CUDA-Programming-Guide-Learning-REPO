// Author: Elin.Liu
// Date: 2022-12-04 10:08:14
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-12-04 10:08:14

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*定义线程id计算函数*/
__global__ void what_is_my_id_2d_A(
    unsigned int *const block_x,
    unsigned int *const block_y,
    unsigned int *const thread,
    unsigned int *const calc_thread,
    unsigned int *const x_thread,
    unsigned int *const y_thread,
    unsigned int *const grid_dimx,
    unsigned int *const block_dimx,
    unsigned int *const grid_dimy,
    unsigned int *const block_dimy)
{
    /*获得线程索引*/
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    /*
        计算线程id
        计算公式：线程ID = ((网格维度x * 块维度x) * 线程idy) + 线程idx(作为x维度上的便宜)
    */
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;
    /*获取线程块的索引*/
    block_x[thread_idx] = blockIdx.x;
    block_y[thread_idx] = blockIdx.y;
    /*获取线程的索引*/
    thread[thread_idx] = threadIdx.x;
    /*计算线程id*/
    calc_thread[thread_idx] = thread_idx;
    /*获取线程的x维度索引*/
    x_thread[thread_idx] = idx;
    /*获取线程的y维度索引*/
    y_thread[thread_idx] = idy;
    /*获取网格维度的X，Y值*/
    grid_dimx[thread_idx] = gridDim.x;
    grid_dimy[thread_idx] = gridDim.y;
    /*获取block_dimy*/
    block_dimx[thread_idx] = blockDim.x;
}

/*定义矩阵宽度以及大小*/
#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16
#define ARRAY_SIZE_IN_BYTES (ARRAY_SIZE_X * ARRAY_SIZE_Y * sizeof(unsigned int))

/*声明CPU端上的各项参数内存*/
unsigned int *cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_warp[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_x_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_y_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int *cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

int main(void)
{
    /*
        初始化dim3常量以配置GPU启动核参数
        注意这里的启动线程块分布是以方形启动，而非矩形启动
    */
    const dim3 thread_square = (16, 8);
    /*注意这里的块的dim3值为2x2*/
    const dim3 block_square = (2, 2);

    /*初始化矩形线程分布启动项*/
    const dim3 thread_rect = (32, 16);
    /*注意这里的块的dim3值为1x4*/
    const dim3 block_rect = (1, 4);

    /*定义一个临时指针用于打印信息*/
    char ch;

    /*定义GPU端上的各项参数内存*/
    unsigned int *gpu_block_x;
    unsigned int *gpu_block_y;
    unsigned int *gpu_thread;
    unsigned int *gpu_warp;
    unsigned int *gpu_calc_thread;
    unsigned int *gpu_x_thread;
    unsigned int *gpu_y_thread;
    unsigned int *gpu_grid_dimx;
    unsigned int *gpu_grid_dimy;
    unsigned int *gpu_block_dimx;

    /*分配GPU端上的各项参数内存*/
    cudaMalloc((void **)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_warp, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_x_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_y_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_grid_dimx, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_grid_dimy, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_block_dimx, ARRAY_SIZE_IN_BYTES);

    /*调用核函数*/
    for (int kernel = 0; kernel < 2; kernel++)
    {
        switch (kernel)
        {
        case 0:
            /*执行矩形配置核函数*/
            what_is_my_id_2d_A<<<block_rect, thread_rect>>>(gpu_block_x, gpu_block_y, gpu_thread, gpu_warp, gpu_calc_thread, gpu_x_thread, gpu_y_thread, gpu_grid_dimx, gpu_grid_dimy, gpu_block_dimx);
            break;
        case 1:
            /*执行方形配置核函数*/
            what_is_my_id_2d_A<<<block_square, thread_square>>>(gpu_block_x, gpu_block_y, gpu_thread, gpu_warp, gpu_calc_thread, gpu_x_thread, gpu_y_thread, gpu_grid_dimx, gpu_grid_dimy, gpu_block_dimx);
            break;
        default:
            exit(1);
            break;
        }

        /*将GPU端上的各项参数内存拷贝到CPU端上*/
        cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_x_thread, gpu_x_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_y_thread, gpu_y_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_dimx, gpu_block_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

        printf("\n kernel %d\n", kernel);
        /*打印结果*/
        for (int y = 0; y < ARRAY_SIZE_Y; y++)
        {
            for (int x = 0; x < ARRAY_SIZE_X; x++)
            {
                printf("CT: %2u Bkx: %1u TID: %2u YTID: %2u XTID: %2u GDX: %1u BDX: %1u GDY: %1u BDY:%1U\n", cpu_calc_thread[y * ARRAY_SIZE_X + x], cpu_block_x[y * ARRAY_SIZE_X + x], cpu_thread[y * ARRAY_SIZE_X + x], cpu_y_thread[y * ARRAY_SIZE_X + x], cpu_x_thread[y * ARRAY_SIZE_X + x], cpu_grid_dimx[y * ARRAY_SIZE_X + x], cpu_block_dimx[y * ARRAY_SIZE_X + x], cpu_grid_dimy[y * ARRAY_SIZE_X + x], cpu_block_y[y * ARRAY_SIZE_X + x]);
            }
            /*每行打印完后按任意键继续*/
            ch = getchar();
        }
        printf("Press any key to continue\n");
        ch = getchar();
    }
    /*释放GPU端上的各项参数内存*/
    cudaFree(gpu_block_x);
    cudaFree(gpu_block_y);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);
    cudaFree(gpu_x_thread);
    cudaFree(gpu_y_thread);
    cudaFree(gpu_grid_dimx);
    cudaFree(gpu_grid_dimy);
    cudaFree(gpu_block_dimx);
}