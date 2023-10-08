#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; ++idx) {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size) {
    // 翻译：为随机数生成不同的种子
    time_t t;  // time_t是一种时间类型
    srand((unsigned int) time(&t));  // time()函数返回当前时间的秒数

    for (int i = 0; i < size; ++i) {  // 生成随机数
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

int main(int argc, char **argv) {
    // 设置默认值
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    // 申请内存
    float *h_A, *h_B, *h_C;
    h_A = (float *) malloc(nBytes);  // malloc()函数用于动态内存分配
    h_B = (float *) malloc(nBytes);  // malloc()函数用于动态内存分配
    h_C = (float *) malloc(nBytes);  // malloc()函数用于动态内存分配

    // 初始化数据
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // 在主机上计算
    sumArraysOnHost(h_A, h_B, h_C, nElem);

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}