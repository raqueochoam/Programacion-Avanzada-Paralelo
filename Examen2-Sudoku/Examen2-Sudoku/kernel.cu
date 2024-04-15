#include <iostream>

const int FILAS = 9;
const int COLUMNAS = 9;

__global__ void chequeoGeneral(char* sudoku, bool* resultado) {
    *resultado = true;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < FILAS; ++i) {
            for (int j = 0; j < COLUMNAS; ++j) {
                if (sudoku[i * COLUMNAS + j] == '\0') {
                    *resultado = false;
                    return;
                }
            }
        }
    }
}

__global__ void chequeoFilas(char* sudoku, bool* resultado) {
    *resultado = true;
    for (int i = 0; i < FILAS; i++) {
        bool fila_valida[FILAS] = { false };
        for (int j = 0; j < COLUMNAS; j++) {
            char elemento = sudoku[i * COLUMNAS + j];
            if (elemento != '.') {
                int valor = elemento - '0';
                if (fila_valida[valor - 1]) {
                    *resultado = false;
                    printf("no es valido, pff\n");
                    return;
                }
                else {
                    fila_valida[valor - 1] = true;
                    printf("todo bien por ahora\n");
                }
            }
        }
    }
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
    bool resultado, * h_resultado;
    h_resultado = &resultado;

    cudaMalloc((void**)&d_sudoku, FILAS * COLUMNAS * sizeof(char));
    cudaMalloc((void**)&d_resultado, sizeof(bool));

    cudaMemcpy(d_sudoku, sudoku, FILAS * COLUMNAS * sizeof(char), cudaMemcpyHostToDevice);

    chequeoGeneral << <1, 1 >> > (d_sudoku, d_resultado);
    cudaMemcpy(h_resultado, d_resultado, sizeof(bool), cudaMemcpyDeviceToHost);

    if (resultado) {
        std::cout << "La matriz es de 9x9." << std::endl;
    }
    else {
        std::cout << "La matriz NO es de 9x9." << std::endl;
    }

    cudaFree(d_sudoku);
    cudaFree(d_resultado);

    return 0;
}
