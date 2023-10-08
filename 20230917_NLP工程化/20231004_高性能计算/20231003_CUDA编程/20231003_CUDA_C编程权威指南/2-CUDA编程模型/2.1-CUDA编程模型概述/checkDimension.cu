// 检查网格和块的索引和维度
# include <cuda_runtime.h>
# include <stdio.h>

__global__ void checkIndex(void) {
    // gridDim表示grid的维度，blockDim表示block的维度，grid维度表示grid中block的数量，block维度表示block中thread的数量
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
           "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z); // printf函数只支持Fermi及以上版本的GPU架构，因此编译的时候需要加上-arch=sm_20编译器选项
}

int main(int argc, char** argv) {
    // 定义全部数据元素
    int nElem = 6;

    // 定义grid和block的结构
    dim3 block(3);  // 表示一个block中有3个线程
    dim3 grid((nElem + block.x - 1) / block.x);  // 表示grid中有2个block

    // 检查grid和block的维度(host端)
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // 检查grid和block的维度(device端)
     checkIndex<<<grid, block>>>();

    // 离开之前重置设备
    cudaDeviceReset();

    return 0;
}