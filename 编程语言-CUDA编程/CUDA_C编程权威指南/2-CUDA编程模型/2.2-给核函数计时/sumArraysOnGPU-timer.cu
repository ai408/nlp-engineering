#include <cuda_runtime.h> // 包含cuda运行时系统的头文件
#include <stdio.h>  // 包含标准输入输出函数的头文件
#include <time.h>  // 包含时间函数的头文件
#include <sys/timeb.h>  // 包含时间函数的头文件
//#define CHECK(call)  // 定义CHECK宏函数


void initialData(float *ip, int size)
{
    // 为随机数生成不同的种子
    time_t t;  // time_t是一种时间类型
    srand((unsigned int) time(&t));  // time()函数返回当前时间的秒数
    for (int i = 0; i < size; i++)  // 生成随机数
    {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;  // rand()函数用于生成随机数
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N)  // 检查结果
{
    double epsilon = 1.0E-8;  // 定义误差范围
    bool match = 1;  // 定义匹配标志
    for (int i = 0; i < N; i++)  // 比较每个元素
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)  // 如果误差超过范围
        {
            match = 0;  // 匹配标志置0
            printf("Arrays do not match!\n");  // 打印提示信息
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);  // 打印不匹配的元素
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");  // 如果匹配，打印提示信息
}


void sumArraysOnHost(float *A, float *B, float *C, const int N)  // 在主机上计算
{
    for (int idx = 0; idx < N; idx++)  // 计算每个元素
    {
        C[idx] = A[idx] + B[idx];  // 计算
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C)  // 在设备上计算
{
//    int i = threadIdx.x;  // 获取线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 获取线程索引
//    printf("threadIdx.x: %d, blockIdx.x: %d, blockDim.x: %d\n", threadIdx.x, blockIdx.x, blockDim.x);  // 打印线程索引
    C[i] = A[i] + B[i];  // 计算
}

struct timeval {  // 定义timeval结构体
    long tv_sec;  // 秒
    long tv_usec; // 微秒
};

int gettimeofday(struct timeval *tp, void *tzp) {  // 定义gettimeofday函数
    struct _timeb timebuffer;  // 定义_timeb结构体
    _ftime(&timebuffer);  // 获取当前时间
    tp->tv_sec = static_cast<long>(timebuffer.time);  // 转换为秒
    tp->tv_usec = timebuffer.millitm * 1000;  // 转换为微秒
    return 0;  // 返回0
}

double cpuSecond() {  // 定义cpuSecond函数
    struct timeval tp;  // 定义timeval结构体
    gettimeofday(&tp, NULL);  // 获取当前时间
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);  // 返回当前时间
}


int main(int argc, char** argv) {
    printf("%s Starting...\n", argv[0]);  // 打印提示信息

    // 设置device
    int dev = 0;  // 定义device
    cudaDeviceProp deviceProp;  // 定义deviceProp结构体
//    CHECK(cudaGetDeviceProperties(&deviceProp, dev));  // 获取deviceProp结构体
    cudaGetDeviceProperties(&deviceProp, dev);  // 获取deviceProp结构体
    printf("Using Device %d: %s\n", dev, deviceProp.name);
//    CHECK(cudaSetDevice(dev));  // 设置device
    cudaSetDevice(dev);  // 设置device

    // 设置vector数据大小
    int nElem = 1 << 24;  // 定义vector大小，左移24位相当于乘以2的24次方
    printf("Vector size %d\n", nElem);  // 打印vector大小

    // 分配主机内存
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;  // 定义主机内存指针
    h_A = (float *) malloc(nBytes);  // 分配主机内存
    h_B = (float *) malloc(nBytes);  // 分配主机内存
    hostRef = (float *) malloc(nBytes);  // 分配主机内存，用于存储host端计算结果
    gpuRef = (float *) malloc(nBytes);  // 分配主机内存，用于存储device端计算结果

    // 定义计时器
    double iStart, iElaps;

    // 初始化主机数据
    iStart = cpuSecond();  // 记录开始时间
    initialData(h_A, nElem);  // 初始化数据
    initialData(h_B, nElem);  // 初始化数据
    iElaps = cpuSecond() - iStart;  // 记录结束时间

    memset(hostRef, 0, nBytes);  // 将hostRef清零
    memset(gpuRef, 0, nBytes);  // 将gpuRef清零

    // 在主机做向量加法
    iStart = cpuSecond();  // 记录开始时间
    sumArraysOnHost(h_A, h_B, hostRef, nElem);  // 在主机上计算
    iElaps = cpuSecond() - iStart;  // 记录结束时间
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);  // 打印执行时间


    // 分配设备全局内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **) &d_A, nBytes);
    cudaMalloc((float **) &d_B, nBytes);
    cudaMalloc((float **) &d_C, nBytes);

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // 在设备端调用kernel
    int iLen = 1024;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    iStart = cpuSecond();  // 记录开始时间
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C);  // 调用kernel
    cudaDeviceSynchronize();  // 同步device
    iElaps = cpuSecond() - iStart;  // 记录结束时间
    printf("sumArraysOnGPU <<<%d, %d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);  // 打印执行时间

    // 拷贝结果到主机
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // 检查结果
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