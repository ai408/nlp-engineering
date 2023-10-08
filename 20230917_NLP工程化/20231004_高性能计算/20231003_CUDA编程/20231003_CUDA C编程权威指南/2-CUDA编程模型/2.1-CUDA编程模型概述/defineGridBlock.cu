#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // 定义全部数据元素
    int cElem = 1024;

    // 定义grid和block结构
    dim3 block(1024);
    dim3 grid((cElem + block.x - 1) / block.x);
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);

    // 重置block
    block.x = 512;
    grid.x = (cElem + block.x - 1) / block.x;
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);

    // 重置block
    block.x = 256;
    grid.x = (cElem + block.x - 1) / block.x;
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);

    // 重置block
    block.x = 128;
    grid.x = (cElem + block.x - 1) / block.x;
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);

    // 离开前重置device
    cudaDeviceReset();
    return 0;
}