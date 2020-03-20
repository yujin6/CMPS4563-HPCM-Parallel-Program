// ***************************************************************************
// Assignment #2
// Name: Yujin Yoshimura
// Parallel Programming Date: March 5, 2020
// ***************************************************************************
// This sequential program demonstrates Matrix Multiplication.
//
// For Turing, use the script on the same directory to compile and run.
// TACC Maverick 2 command to compile:
// gcc YujinYoshimuraSequential.c -o YujinYoshimuraSequential_Exe
// ***************************************************************************

#include <stdio.h>
#include "timer.h"

const int ROW = 32;
const int COL = 32;

void cross_product(int* a, int* b, int* c);

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
  double start, finish, elapsed;

  // Initialize matrix A, B and C
  for (i = 0; i < ROW; i++) {
    for (j = 0; j < COL; j++) {
      matrix_a[i * COL + j] = i;
      matrix_b[i * COL + j] = COL - i - 1;
      matrix_c[i * COL + j] = 0;
    }
  }

  GET_TIME(start);
  cross_product(matrix_a, matrix_b, matrix_c);
  GET_TIME(finish);
  elapsed = finish - start;

  for (i = 0; i < ROW; i++) {
    for (j = 0; j < COL; j++) {
      sum += matrix_c[i * COL + j];
    }
  }

  printf("The summation of all the elements is = %d\n", sum);
  printf("Elapsed time = %f milliseconds.\n", elapsed * 1000);

  return 0;
}

// ***************************************************************************
// Function Name: cross_product
// Parameters: int*, int*, int*
// Return: void
// Description: Returns the cross product of two matrices.
// ***************************************************************************
void cross_product(int* a, int* b, int* c) {
  int i, j, k;

  for (i = 0; i < ROW; i++) {
    for (j = 0; j < COL; j++) {
      c[i * COL + j] = 0;
      for (k = 0; k < ROW; k++) {
        c[i * COL + j] += a[i * COL + k] * b[k * COL + j];
      }
    }
  }
}
