// ***************************************************************************
// In Class Activity
// Name: Yujin Yoshimura
// Parallel Programming Date: April 8, 2020
// ***************************************************************************
// This sequential program demonstrates Matrix Multiplication.
//
// For Turing, use the script on the same directory to compile and run.
// TACC Maverick 2 command to compile:
// nvcc YujinYoshimuraSharedMemory.cu -o YujinYoshimuraSharedMemory_Exe
// ***************************************************************************

#include <cuda.h>
#include <stdio.h>

const int ROW = 1024;

// ***************************************************************************
// Function Name: product
// Parameters: int*, int*, int*
// Return: void
// Description: Returns the cross product of two matrices.
// ***************************************************************************
__global__
void product(int* a, int* b, int* c) {
  int i = threadIdx.x;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k;
  // Allocate shared memory
  __shared__ int as[ROW];
  __shared__ int bs[ROW];
  __shared__ int cs[ROW];

  for (k = 0; k < 5; k++) {
    // Copy from global memory to shared memory
    as[i] = a[j + ROW * k * 2];
    bs[i] = b[j + ROW * k * 2];
    // Do the math
    cs[i] = as[i] * bs[i];
    // Copy from shared memory to global memory
    c[j + ROW * k * 2] = cs[i];
  }
}

// ***************************************************************************
// Function Name: main
// Parameters: int, char**
// Return: int
// Description: Main function of the program.
// ***************************************************************************
int main(int argc, char **argv) {
  int matrix_a[ROW * 10];
  int matrix_b[ROW * 10];
  int matrix_c[ROW * 10];
  int i, sum = 0;
  int* ad;
  int* bd;
  int* cd;
  const int isize = ROW * 10 * sizeof(int);

  // Initialize matrix A, B and C
  for (i = 0; i < ROW * 10; i++) {
    matrix_a[i] = 2;
    matrix_b[i] = 20;
    matrix_c[i] = 0;
  }

  // Allocate memory and copy matrices to global memory
  cudaMalloc( (void**)&ad, isize );
 	cudaMalloc( (void**)&bd, isize );
  cudaMalloc( (void**)&cd, isize );
 	cudaMemcpy( ad, matrix_a, isize, cudaMemcpyHostToDevice );
  cudaMemcpy( bd, matrix_b, isize, cudaMemcpyHostToDevice );
  cudaMemcpy( cd, matrix_c, isize, cudaMemcpyHostToDevice );

  dim3 dimGrid( 2 , 1 );
	dim3 dimBlock( ROW , 1 );

  product<<<dimGrid, dimBlock>>>(ad, bd, cd);

  // Copy matrix to memory and free global memory
  cudaMemcpy( matrix_c, cd, isize, cudaMemcpyDeviceToHost );
 	cudaFree( ad );
  cudaFree( bd );
  cudaFree( cd );

   for (i = 0; i < ROW * 10; i++) {
      sum += matrix_c[i];
  }

  printf("The summation of all the elements is = %d\n", sum);

 	return EXIT_SUCCESS;
}
