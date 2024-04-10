#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

//Paritr en dos los datos
int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return (i + 1);
}

__device__ int partition2(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return (i + 1);
}

// Función recursiva para Quick Sort
void quickSort(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Función recursiva para Quick Sort
__global__ void quickSort2(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition2(arr, low, high);
        quickSort2(arr, low, pi - 1);
        quickSort2(arr, pi + 1, high);
    }
}

int main() {
    int* a_cpu;

    int* a_device;
  
    const int data_count = 10000;
    int* data = (int*)malloc(data_count * sizeof(int));

    // Generar datos aleatorios
    for (int i = 0; i < data_count; i++) {
        data[i] = rand() % 1000;
    }
   
    a_cpu = (int*)malloc(data);

    // Asignación de memoria en el dispositivo
    cudaMalloc((void**)&a_device, data);

    // Transferir datos del host al dispositivo
    cudaMemcpy(a_device, a_cpu, data, cudaMemcpyHostToDevice);
  
    //Lanzar Kernel, que ahporita es funcion
    //quickSort(data, 0, data_count - 1);

    quickSort2 << <data, 0, data_count-1>> > ();

    // Imprimir datos ordenados
    printf("Datos ordenados:\n");
    for (int i = 0; i < data_count; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    // Transferir datos del dispositivo al host
    cudaMemcpy(a_cpu, a_device, data, cudaMemcpyDeviceToHost);

    // Liberar memoria en el host
    free(a_cpu);


    // Liberar memoria en el dispositivo
    cudaFree(a_device);


    // Restablecer el dispositivo CUDA
    cudaDeviceReset();

    return 0;
}