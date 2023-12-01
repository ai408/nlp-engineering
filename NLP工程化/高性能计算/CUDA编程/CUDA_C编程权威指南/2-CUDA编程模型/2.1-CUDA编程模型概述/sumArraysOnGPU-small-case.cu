#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)
//{
//    const cudaError_t error = call;
//    if (error != cudaSuccess)
//    {
//        printf("Error: %s:%d, ", __FILE__, __LINE__);
//        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
//        exit(1);
//    }
//}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
//    int i = threadIdx.x;  // 获取线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 获取线程索引
    printf("threadIdx.x: %d, blockIdx.x: %d, blockDim.x: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
    C[i] = A[i] + B[i];  // 计算
}

int main(int argc, char** argv) {
    printf("%s Starting...\n", argv[0]);

    // 设置设备
    int dev = 0;
    cudaSetDevice(dev);

    // 设置vectors数据大小
    int nElem = 32;
    printf("Vector size %d\n", nElem);

    // 分配主机内存
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;  // 定义主机内存指针
    h_A = (float *) malloc(nBytes);  // 分配主机内存
    h_B = (float *) malloc(nBytes);  // 分配主机内存
    hostRef = (float *) malloc(nBytes);  // 分配主机内存，用于存储host端计算结果
    gpuRef = (float *) malloc(nBytes);  // 分配主机内存，用于存储device端计算结果

    // 初始化主机数据
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);  // 将hostRef清零
    memset(gpuRef, 0, nBytes);  // 将gpuRef清零

    // 分配设备全局内存
    float *d_A, *d_B, *d_C;  // 定义设备内存指针
    cudaMalloc((float **) &d_A, nBytes);  // 分配设备内存
    cudaMalloc((float **) &d_B, nBytes);  // 分配设备内存
    cudaMalloc((float **) &d_C, nBytes);  // 分配设备内存

    // 从主机内存拷贝数据到设备内存
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);  // d_A表示目标地址，h_A表示源地址，nBytes表示拷贝字节数，cudaMemcpyHostToDevice表示拷贝方向
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);  // d_B表示目标地址，h_B表示源地址，nBytes表示拷贝字节数，cudaMemcpyHostToDevice表示拷贝方向

    // 在host端调用kernel
    dim3 block(nElem);  // 定义block维度
    dim3 grid(nElem / block.x);  // 定义grid维度

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C);  // 调用kernel，<<<grid, block>>>表示执行配置，d_A, d_B, d_C表示kernel参数
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);  // 打印执行配置

    // 拷贝device结果到host内存
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);  // gpuRef表示目标地址，d_C表示源地址，nBytes表示拷贝字节数，cudaMemcpyDeviceToHost表示拷贝方向

    // 在host端计算结果
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // 检查device结果
    checkResult(hostRef, gpuRef, nElem);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}