// ***************************************************************************
// Assignment #4
// Name: Yujin Yoshimura
// Parallel Programming Date: November 7, 2019
// Version 1.0
// Instructor: Dr. Eduardo Colmenares
// ***************************************************************************
// This sequential program performs addition, subtraction, multiplication, and
// division of two arrays, element-wise.
//
// TACC command to compile:
// mpicc YoshimuraA4NoThreads.c -o YoshimuraA4NoThreads_Exe
// ***************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

int* create_iarray(int size);
double* create_darray(int size);
void addition(int* sum, int* a, int* b);
void subtraction(int* difference, int* a, int* b);
void multiplication(int* product, int* a, int* b);
void division(double* quotient, int* a, int* b);

const int ARRAYSIZE = 1000000;

int main(int argc, char **argv) {
  int* a = create_iarray(ARRAYSIZE);
  int* b = create_iarray(ARRAYSIZE);
  int* s = create_iarray(ARRAYSIZE);
  int* d = create_iarray(ARRAYSIZE);
  int* p = create_iarray(ARRAYSIZE);
  double* q = create_darray(ARRAYSIZE);
  double start, finish;

  // initialize array
  for (int i = 0; i < ARRAYSIZE; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  // perform fundamental operations
  GET_TIME(start);
  addition(s, a, b);
  subtraction(d, a, b);
  multiplication(p, a, b);
  division(q, a, b);
  GET_TIME(finish);

  printf("Sum is %d,\n", s[ARRAYSIZE - 1]);
  printf("Difference is %d,\n", d[ARRAYSIZE - 1]);
  printf("Product is %d,\n", p[ARRAYSIZE - 1]);
  printf("Quotient is %.1f,\n", q[ARRAYSIZE - 1]);
  printf("Elapsed time = %f milliseconds.\n", (finish - start) * 1000);

  return 0;
}

// ***************************************************************************
// Function Name: create_iarray
// Parameters: int
// Return: int*
// Description: Creates an array of int with a given size.
// ***************************************************************************
int* create_iarray(int size) {
  int *a = (int *) calloc(size, sizeof(int));
  return a;
}

// ***************************************************************************
// Function Name: create_darray
// Parameters: int
// Return: double*
// Description: Creates an array of double with a given size.
// ***************************************************************************
double* create_darray(int size) {
  double *a = (double *) calloc(size, sizeof(double));
  return a;
}

// ***************************************************************************
// Function Name: addition
// Parameters: int*, int*, int*
// Return: void
// Description: Adds two arrays of the same size element-wise.
// ***************************************************************************
void addition(int* sum, int* a, int* b) {
  for (int i = 0; i < ARRAYSIZE; i++) {
    sum[i] = a[i] + b[i];
  }
}

// ***************************************************************************
// Function Name: subtraction
// Parameters: int*, int*, int*
// Return: void
// Description: Subtracts two arrays of the same size element-wise.
// ***************************************************************************
void subtraction(int* difference, int* a, int* b) {
  for (int i = 0; i < ARRAYSIZE; i++) {
    difference[i] = a[i] - b[i];
  }
}

// ***************************************************************************
// Function Name: multiplication
// Parameters: int*, int*, int*
// Return: void
// Description: Multiplies two arrays of the same size element-wise.
// ***************************************************************************
void multiplication(int* product, int* a, int* b) {
  for (int i = 0; i < ARRAYSIZE; i++) {
    product[i] = a[i] * b[i];
  }
}

// ***************************************************************************
// Function Name: division
// Parameters: double*, int*, int*
// Return: void
// Description: Divides two arrays of the same size element-wise.
// ***************************************************************************
void division(double* quotient, int* a, int* b) {
  for (int i = 0; i < ARRAYSIZE; i++) {
    quotient[i] = (double)a[i] / (double)b[i];
  }
}
