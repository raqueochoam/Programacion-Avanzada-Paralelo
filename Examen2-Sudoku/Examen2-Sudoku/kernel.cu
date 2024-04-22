#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib> // For std::rand() and std::srand()
#include <ctime>   // For std::time()
#include <chrono>  // For timing measurements
#include "device_launch_parameters.h"  // For using blockDim, threadIdx, etc.

const int FILAS = 9;
const int COLUMNAS = 9;

__global__ void chequeoFilas(char* sudoku, bool* resultado) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    if (fila < FILAS) {
        bool fila_valida[FILAS] = { false };
        for (int j = 0; j < COLUMNAS; j++) {
            char elemento = sudoku[fila * COLUMNAS + j];
            if (elemento != '.') {
                int valor = elemento - '0';
                if (fila_valida[valor - 1]) {
                    printf("peto en una fila\n");
                    resultado[fila] = false;
                    return;
                }
                else {
                    fila_valida[valor - 1] = true;
                    
                }
            }
        }
    }
}

__global__ void chequeoColumnas(char* sudoku, bool* resultado) {
    int columna = blockIdx.x * blockDim.x + threadIdx.x;
    if (columna < COLUMNAS) {
        bool columna_valida[COLUMNAS] = { false };
        for (int i = 0; i < FILAS; i++) {
            char elemento = sudoku[i * COLUMNAS + columna];
            if (elemento != '.') {
                int valor = elemento - '0';
                if (columna_valida[valor - 1]) {
                    printf("peto en una columna\n");
                    resultado[columna + FILAS] = false;
                    return;
                }
                else {
                    columna_valida[valor - 1] = true;
                }
            }
        }
    }
}

__global__ void chequeoSubcuadros(char* sudoku, bool* resultado) {
    int subcuadro_id = blockIdx.x * blockDim.x + threadIdx.x;
    int inicio_fila = (subcuadro_id / 3) * 3;
    int inicio_columna = (subcuadro_id % 3) * 3;

    bool subcuadro_valido[9] = { false };

    for (int i = inicio_fila; i < inicio_fila + 3; ++i) {
        for (int j = inicio_columna; j < inicio_columna + 3; ++j) {
            char elemento = sudoku[i * COLUMNAS + j];
            if (elemento != '.') {
                int valor = elemento - '0';
                if (subcuadro_valido[valor - 1]) {
                    printf("peto en subcuadro\n");
                    resultado[subcuadro_id] = false;
                    return;
                }
                else {
                    subcuadro_valido[valor - 1] = true;
                }
            }
        }
    }
}

// Función para verificar si un valor es válido en una posición dada del sudoku
bool esValido(char* sudoku, int fila, int columna, char valor) {
    // Verifica la fila y la columna
    for (int i = 0; i < 9; ++i) {
        if (sudoku[fila * COLUMNAS + i] == valor || sudoku[i * COLUMNAS + columna] == valor) {
            return false;
        }
    }

    // Verifica el subcuadro
    int inicio_fila = (fila / 3) * 3;
    int inicio_columna = (columna / 3) * 3;
    for (int i = inicio_fila; i < inicio_fila + 3; ++i) {
        for (int j = inicio_columna; j < inicio_columna + 3; ++j) {
            if (sudoku[i * COLUMNAS + j] == valor) {
                return false;
            }
        }
    }

    return true;
}

// Función de backtracking para resolver el sudoku
bool resolverSudoku(char* sudoku, int fila, int columna) {
    if (fila == 9) {
        return true; // Se ha resuelto todo el sudoku
    }

    if (sudoku[fila * COLUMNAS + columna] != '.') {
        // Si la celda ya está llena, pasa a la siguiente
        if (columna == 8) {
            return resolverSudoku(sudoku, fila + 1, 0);
        }
        else {
            return resolverSudoku(sudoku, fila, columna + 1);
        }
    }

    for (char c = '1'; c <= '9'; ++c) {
        if (esValido(sudoku, fila, columna, c)) {
            sudoku[fila * COLUMNAS + columna] = c;
            if (columna == 8) {
                if (resolverSudoku(sudoku, fila + 1, 0)) {
                    return true;
                }
            }
            else {
                if (resolverSudoku(sudoku, fila, columna + 1)) {
                    return true;
                }
            }
            sudoku[fila * COLUMNAS + columna] = '.'; // Si no se encontró solución, vuelve a poner la celda como vacía
        }
    }

    return false; // No se encontró ninguna solución
}

int main() {
    char sudoku[FILAS][COLUMNAS] = {
        {'5', '3', '.', '.', '7', '.', '.', '.', '.'},
        {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
        {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
        {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
        {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
        {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
        {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
        {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
        {'.', '.', '.', '.', '8', '.', '.', '7', '9'}
    };

    char* d_sudoku;
    bool* d_resultado;
    bool resultado[FILAS + COLUMNAS];
    bool* h_resultado = new bool[FILAS + COLUMNAS]; 
    for (int i = 0; i < FILAS + COLUMNAS; ++i) {
        h_resultado[i] = true; 
    }

    std::srand(static_cast<unsigned>(std::time(nullptr))); // Seed srand with current time

    cudaMalloc((void**)&d_sudoku, FILAS * COLUMNAS * sizeof(char));
    cudaMalloc((void**)&d_resultado, (FILAS + COLUMNAS) * sizeof(bool)); // Reservar memoria en el dispositivo para los resultados

    cudaMemcpy(d_resultado, h_resultado, (FILAS + COLUMNAS) * sizeof(bool), cudaMemcpyHostToDevice); // Copiar los resultados al dispositivo
    cudaMemcpy(d_sudoku, sudoku, FILAS * COLUMNAS * sizeof(char), cudaMemcpyHostToDevice);

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    //ejecutar kernel
    chequeoFilas << <1, FILAS >> > (d_sudoku, d_resultado);
    chequeoColumnas << <1, COLUMNAS >> > (d_sudoku, d_resultado);
    chequeoSubcuadros << <1, 9 >> > (d_sudoku, d_resultado);

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calcular la duración en milisegundos
    std::chrono::duration<float, std::milli> duration_ms = end - start;

    // Imprimir el tiempo de ejecución
    std::cout << "Tiempo de ejecucion: " << duration_ms.count() << " ms" << std::endl;

    cudaMemcpy(h_resultado, d_resultado, (FILAS + COLUMNAS) * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_sudoku);
    cudaFree(d_resultado);

    bool sudoku_valido = true;
    for (int i = 0; i < FILAS + COLUMNAS; ++i) {
        if (!h_resultado[i]) {
            sudoku_valido = false;
            break;
        }
    }

    delete[] h_resultado; // Liberar la memoria del array en el host

    if (sudoku_valido) {
        std::cout << "El sudoku es valido." << std::endl;

        // Resolver el Sudoku
        if (resolverSudoku(sudoku[0], 0, 0)) {
            std::cout << "Sudoku resuelto:" << std::endl;
            for (int i = 0; i < FILAS; ++i) {
                for (int j = 0; j < COLUMNAS; ++j) {
                    std::cout << sudoku[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
        else {
            std::cout << "No se pudo resolver el Sudoku." << std::endl;
        }
    }
    else {
        std::cout << "El sudoku NO es valido." << std::endl;
    }

    return 0;
}