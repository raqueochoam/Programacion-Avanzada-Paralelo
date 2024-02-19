#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define N 10000

__global__ void vectorSum(float* a, float* b, float* c, float* result)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    int threadId = id_x + id_y * gridDim.x * blockDim.x + id_z * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    if (threadId < N)
    {
        result[threadId] = a[threadId] + b[threadId] + c[threadId];
    }
}

int main()
{
    // Tamaño de los vectores
    int vectorSize = N * sizeof(float);

    // Vectores en el host
    float* h_a = (float*)malloc(vectorSize);
    float* h_b = (float*)malloc(vectorSize);
    float* h_c = (float*)malloc(vectorSize);
    float* h_result = (float*)malloc(vectorSize);

    // Inicialización de los vectores
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = i;
        h_b[i] = 2 * i;
        h_c[i] = 3 * i;
    }

    // Vectores en el dispositivo
    float* d_a, * d_b, * d_c, * d_result;
    cudaMalloc((void**)&d_a, vectorSize);
    cudaMalloc((void**)&d_b, vectorSize);
    cudaMalloc((void**)&d_c, vectorSize);
    cudaMalloc((void**)&d_result, vectorSize);

    // Copiar datos del host al dispositivo
    cudaMemcpy(d_a, h_a, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, vectorSize, cudaMemcpyHostToDevice);

    // Configuración de dimensiones de bloque y grid
    dim3 blockSize(8, 8, 8); // 8 hilos en cada dimensión
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (N + blockSize.z - 1) / blockSize.z);

    // Lanzar el kernel
    vectorSum << <gridSize, blockSize >> > (d_a, d_b, d_c, d_result);

    // Sincronizar para asegurar que el kernel termina antes de copiar los resultados de vuelta al host
    cudaDeviceSynchronize();

    // Copiar resultados de vuelta al host
    cudaMemcpy(h_result, d_result, vectorSize, cudaMemcpyDeviceToHost);

    // Imprimir algunos resultados para verificar
    for (int i = 0; i < 10; ++i)
    {
        printf("Resultado[%d] = %f\n", i, h_result[i]);
    }

    // Liberar memoria en el dispositivo
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_result);

    // Liberar memoria en el host
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_result);

    return 0;
}
