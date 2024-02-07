#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_hello_cuda()
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;

    printf("DEVICE ThreadIdx.x: %d, blockId.x: %d \n", tidx, blockIdx.x);
    printf("DEVICE ThreadIdx.y: %d, blockId.y: %d \n", tidy, blockIdx.y);
    printf("DEVICE ThreadIdx.z: %d, blockId.z: %d \n", tidz, blockIdx.z);
}

int main()
{
    // Inicialización
    dim3 blockSize(4, 4, 4);
    dim3 gridSize(2, 2, 2);

    int* c_cpu;
    int* a_cpu;
    int* b_cpu;

    int* c_device;
    int* a_device;
    int* b_device;

    const int data_count = 10000;
    const int data_size = data_count * sizeof(int);

    c_cpu = (int*)malloc(data_size);
    a_cpu = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);

    // Asignación de memoria en el dispositivo
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);

    // Transferir datos del host al dispositivo
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);

    // Lanzar el kernel
    print_hello_cuda << <gridSize, blockSize >> > ();

    // Transferir datos del dispositivo al host
    cudaMemcpy(c_cpu, c_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_cpu, a_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_device, data_size, cudaMemcpyDeviceToHost);

    // Liberar memoria en el host
    free(c_cpu);
    free(a_cpu);
    free(b_cpu);

    // Liberar memoria en el dispositivo
    cudaFree(c_device);
    cudaFree(a_device);
    cudaFree(b_device);

    // Restablecer el dispositivo CUDA
    cudaDeviceReset();

    return 0;
}
