﻿
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <iostream>


using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__)}


inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: $s $s $d \n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


void queryDevice() {
    int d_Count = 0;

    cudaGetDeviceCount(&d_Count);

    if (d_Count == 0) {
        cout << ("No cuda device found: \n\r");
    }

    cudaDeviceProp(prop);

    for (int devNo = 0; devNo < d_Count; devNo++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devNo);
        printf("Device number: %d\n");
        printf("    Device Name: %s\n", prop.name);
        printf("     No. of Multiprocessors:                     %d\n", prop.multiProcessorCount);
        printf("     Computer Capability:                        %d,%d\n", prop.major, prop.minor);
        printf("     Memory Clock Rate (KHz):                    %d\n", prop.memoryClockRate);
        printf("     Memory Bus Rate (bits):                     %d\n", prop.memoryBusWidth);
        printf("     Peak Memory Bandwith (GB/s):                %8.2f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("     Total Amount of Global Memory:              %dKB\n", prop.totalGlobalMem / 1024);
        printf("     Total Amount of Const  Memory:              %dKB\n", prop.totalConstMem / 1024);
        printf("     Total of Shared Memory per block:           %dKB\n", prop.sharedMemPerBlock / 1024);
        printf("     Total of Shared Memory per mp:              %dKB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("     Warp Size:                                  %d\n", prop.warpSize);
        printf("     Max. theards per block:                     %d\n", prop.maxThreadsPerBlock);
        printf("     Max. theards per MP:                        %d\n", prop.maxThreadsPerMultiProcessor);
        printf("     Maximun Grid Size:                         (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("     Maximun block dimesion:                    (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }
}

int main()
{
    queryDevice();

    return 0;
}