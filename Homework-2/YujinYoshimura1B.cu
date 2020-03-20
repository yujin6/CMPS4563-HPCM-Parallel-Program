// ***************************************************************************
// Assignment #2
// Name: Yujin Yoshimura
// Parallel Programming Date: March 5, 2020
// ***************************************************************************
// This sequential program demonstrates Matrix Multiplication.
//
// For Turing, use the script on the same directory to compile and run.
// TACC Maverick 2 command to compile:
// gcc YujinYoshimura1B.cu -o YujinYoshimura1B_Exe
// ***************************************************************************

#include <cuda.h>
#include <stdio.h>

const int ROW = 32;
const int COL = 32;

// ***************************************************************************
// Function Name: cross_product
// Parameters: int*, int*, int*
// Return: void
// Description: Returns the cross product of two matrices.
// ***************************************************************************
__global__
void cross_product(int* a, int* b, int* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k;

  c[i * COL + j] = 0;
  for (k = 0; k < ROW; k++) {
    c[i * COL + j] += a[i * COL + k] * b[k * COL + j];
  }
}

// ***************************************************************************
// Function Name: main
// Parameters: int, char**
// Return: int
// Description: Main function of the program.
// ***************************************************************************
int main(int argc, char **argv) {
  int matrix_a[ROW * COL];
  int matrix_b[ROW * COL];
  int matrix_c[ROW * COL];
  int i, j, sum = 0;
  int* ad;
  int* bd;
  int* cd;
  const int isize = ROW * COL * sizeof(int);
  float elapsed;
  cudaEvent_t start, stop;

  // Create CUDA Events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Initialize matrix A, B and C
  for (i = 0; i < ROW; i++) {
    for (j = 0; j < COL; j++) {
      matrix_a[i * COL + j] = i;
      matrix_b[i * COL + j] = COL - i - 1;
      matrix_c[i * COL + j] = 0;
    }
  }

  // Allocate memory and copy matrices to global memory
  cudaMalloc( (void**)&ad, isize );
 	cudaMalloc( (void**)&bd, isize );
  cudaMalloc( (void**)&cd, isize );
 	cudaMemcpy( ad, matrix_a, isize, cudaMemcpyHostToDevice );
  cudaMemcpy( bd, matrix_b, isize, cudaMemcpyHostToDevice );
  cudaMemcpy( cd, matrix_c, isize, cudaMemcpyHostToDevice );

  dim3 dimGrid( 1 , 1 );
	dim3 dimBlock( COL , ROW );

  cudaEventRecord(start);
  cross_product<<<dimGrid, dimBlock>>>(ad, bd, cd);
  cudaEventRecord(stop);

  // Copy matrix to memory, time and free global memory
  cudaMemcpy( matrix_c, cd, isize, cudaMemcpyDeviceToHost );
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
 	cudaFree( ad );
  cudaFree( bd );
  cudaFree( cd );

   for (i = 0; i < ROW; i++) {
    for (j = 0; j < COL; j++) {
      sum += matrix_c[i * COL + j];
    }
  }

  printf("The summation of all the elements is = %d\n", sum);
  printf("Elapsed time = %f milliseconds.\n", elapsed);

 	return EXIT_SUCCESS;
}
