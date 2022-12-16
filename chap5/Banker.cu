// Author: Elin.Liu
// Date: 2022-11-26 14:09:34
// Last Modified by:   Elin.Liu
// Last Modified time: 2022-11-26 14:09:34

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define MAX_PROCESS 4
#define MAX_RESOURCE 4

const int ready = 0;
const int done = 1;

// 初始化结构体
struct Process
{
    int ID;
    int MAX_REQUIRE[MAX_RESOURCE];
    int ALLOCATED[MAX_RESOURCE];
    int NEED[MAX_RESOURCE];
    int STATUS;
};

// 定义cuRand内核初始化函数
__global__ void initKernel(curandState *state, unsigned long seed)
{
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

// 定义初始化资源函数
__global__ void InitResouorce(int r[MAX_RESOURCE], curandState *state)
{
    int id = threadIdx.x;
    r[id] = curand(&state[id]) % 10 + 10;
}

// 定义初始化进程函数
__global__ void InitProcess(struct Process p[MAX_PROCESS], curandState *state)
{
    int i = threadIdx.x;
    curandState localState = state[i];
    p[i].ID = i;
    p[i].MAX_REQUIRE[0] = curand(&localState) % 5 + 2;
    p[i].MAX_REQUIRE[1] = curand(&localState) % 5 + 2;
    p[i].MAX_REQUIRE[2] = curand(&localState) % 5 + 2;
    p[i].MAX_REQUIRE[3] = curand(&localState) % 5 + 2;
    p[i].ALLOCATED[0] = curand(&localState) % 2 + 1;
    p[i].ALLOCATED[1] = curand(&localState) % 2 + 1;
    p[i].ALLOCATED[2] = curand(&localState) % 2 + 1;
    p[i].ALLOCATED[3] = curand(&localState) % 2 + 1;
    p[i].NEED[0] = p[i].MAX_REQUIRE[0] - p[i].ALLOCATED[0];
    p[i].NEED[1] = p[i].MAX_REQUIRE[1] - p[i].ALLOCATED[1];
    p[i].NEED[2] = p[i].MAX_REQUIRE[2] - p[i].ALLOCATED[2];
    p[i].NEED[3] = p[i].MAX_REQUIRE[3] - p[i].ALLOCATED[3];
    p[i].STATUS = ready;
    __syncthreads();
}

// 定义加载系统资源函数
__global__ void LoadSystemResource(int r[MAX_RESOURCE], struct Process *p)
{
    int i = threadIdx.x;
    for (int j = 0; j < MAX_RESOURCE; j++)
    {
        r[j] = r[j] - p[i].ALLOCATED[j];
    }
    __syncthreads();
}

// 定义进程执行函数
__global__ void ProcessRun(int r[MAX_RESOURCE], struct Process *p)
{
    int i = threadIdx.x;
    Process localProcess = p[i];
    for (int j = 0; j < MAX_RESOURCE; j++)
    {
        if (localProcess.NEED[j] > r[j])
        {
            printf("[%p]: Process %d Running Failed,due to system resource is not enough.", &localProcess, localProcess.ID);
        }
        else
        {
            // 回收资源
            r[j] = r[j] + localProcess.ALLOCATED[j];
        }
    }
    localProcess.STATUS = done;
    p[i] = localProcess;
    printf("[%p]: Process %d Running Success.\n", &localProcess, localProcess.ID);
    __syncthreads();
}

// 定义主函数
int main(void)
{
    // 初始化cuRand状态
    curandState *devState;
    cudaMalloc((void **)&devState, MAX_PROCESS * sizeof(curandState));
    srand(time(NULL));
    initKernel<<<1, MAX_PROCESS>>>(devState, rand());
    cudaDeviceSynchronize();
    // 初始化CPU资源指针
    int *r = (int *)malloc(MAX_RESOURCE * sizeof(int));
    // 初始化GPU资源指针
    int *devR;
    cudaMalloc((void **)&devR, MAX_RESOURCE * sizeof(int));
    // 执行初始化资源函数
    InitResouorce<<<1, MAX_RESOURCE>>>(devR, devState);
    cudaDeviceSynchronize();
    // 将GPU资源指针拷贝到CPU资源指针
    cudaMemcpy(r, devR, MAX_RESOURCE * sizeof(int), cudaMemcpyDeviceToHost);
    // 拷贝一份资源状态
    int *MemoryInitState = r;
    // 打印系统资源
    printf("[%p]Current System Resource is: ", &r);
    for (int i = 0; i < MAX_RESOURCE; i++)
    {
        printf("%d ", r[i]);
    }
    printf("\n");
    // 初始化CPU进程指针
    struct Process *p = (struct Process *)malloc(MAX_PROCESS * sizeof(struct Process));
    // 初始化GPU进程指针
    struct Process *devP;
    cudaMalloc((void **)&devP, MAX_PROCESS * sizeof(struct Process));
    // 执行初始化进程函数
    InitProcess<<<1, MAX_PROCESS>>>(devP, devState);
    cudaDeviceSynchronize();
    // 将GPU进程指针拷贝到CPU进程指针
    cudaMemcpy(p, devP, MAX_PROCESS * sizeof(struct Process), cudaMemcpyDeviceToHost);
    // 打印进程信息
    for (int i = 0; i < MAX_PROCESS; i++)
    {
        printf("[%p]:Process %d Maximum required: ", &r[i], p[i].ID);
        for (int j = 0; j < MAX_RESOURCE; j++)
        {
            printf("%d ", p[i].MAX_REQUIRE[j]);
        }
        printf("\tAllocated with : ");
        for (int j = 0; j < MAX_RESOURCE; j++)
        {
            printf("%d ", p[i].ALLOCATED[j]);
        }
        printf("\tStill need:");
        for (int j = 0; j < MAX_RESOURCE; j++)
        {
            printf("%d ", p[i].NEED[j]);
        }
        printf("\n");
    }
    // 执行加载系统资源函数
    printf("[INFO] PreAllocating Resources to Process...\n");
    LoadSystemResource<<<MAX_PROCESS, MAX_RESOURCE>>>(devR, devP);
    cudaDeviceSynchronize();
    // 将GPU资源指针拷贝到CPU资源指针
    cudaMemcpy(r, devR, MAX_RESOURCE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("[%p]Current System Resource is:", &r);
    // 打印系统资源
    for (int i = 0; i < MAX_RESOURCE; i++)
    {
        if (r[i] < 0)
        {
            printf("[ERROR] System Resources doesn`t enough to hold!\n");
            exit(-1);
        }
        else
        {
            printf(" %d", r[i]);
        }
    }
    printf("\n");
    // 执行进程执行函数
    ProcessRun<<<1, MAX_PROCESS>>>(devR, devP);
    cudaDeviceSynchronize();
    // 将GPU资源指针拷贝到CPU资源指针
    cudaMemcpy(r, devR, MAX_RESOURCE * sizeof(int), cudaMemcpyDeviceToHost);
    // 打印系统资源
    printf("[INFO] System Resources after Process executed: ");
    for (int i = 0; i < MAX_RESOURCE; i++)
    {
        printf("%d ", r[i]);
    }
    printf("\n");
    // 检查系统资源是否回收完毕
    for (int i = 0; i < MAX_RESOURCE; i++)
    {
        if (r[i] != MemoryInitState[i])
        {
            printf("[ERROR] System Resources doesn`t recover!\n");
            exit(-1);
        }
    }
    printf("[INFO] System Resources recover!\n");
    // 释放资源
    cudaFree(devR);
    cudaFree(devP);
    cudaFree(devState);
    free(r);
    free(p);
    return 0;
}