#include "cuda_runtime.h" // CUDA运行时API
#include <stdio.h> // 标准输入输出

__global__ void helloFromGPU(void) // GPU核函数
{
	printf("Hello World from GPU!\n"); //输出Hello World from GPU!
}

int main(void) // 主函数
{
	// hello from cpu
	printf("Hello World from GPU!\n"); //CPU主机端输出Hello World from CPU!

	helloFromGPU<<<1,10>>>(); // 调用GPU核函数，10个线程块，1表示每个grid中只有1个block，10表示每个block中有10个线程
	cudaDeviceReset(); // 重置当前设备上的所有资源状态，清空当前设备上的所有内存

	return 0;
}