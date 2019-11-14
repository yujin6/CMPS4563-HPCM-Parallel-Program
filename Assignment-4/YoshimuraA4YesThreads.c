// ***************************************************************************
// Assignment #4
// Name: Yujin Yoshimura
// Parallel Programming Date: November 7, 2019
// Version: 1.0
// Instructor: Dr. Eduardo Colmenares
// ***************************************************************************
// This parallel program performs addition, subtraction, multiplication, and
// division of two arrays, element-wise.
//
// TACC command to compile:
// mpicc YoshimuraA4YesThreads.c -pthread -o YoshimuraA4YesThreads_Exe
// ***************************************************************************

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#define NUM_THREADS 4

void *fundamental_operations(void *threadid);
int* create_iarray(int size);
double* create_darray(int size);
void addition(int* sum, int* a, int* b);
void subtraction(int* difference, int* a, int* b);
void multiplication(int* product, int* a, int* b);
void division(double* quotient, int* a, int* b);

const int ARRAYSIZE = 1000000;

int* a;
int* b;
int* s;
int* d;
int* p;
double* q;

int main(int argc, char **argv) {
  pthread_t threads[NUM_THREADS];
  int rc;
  double start, finish;
  a = create_iarray(ARRAYSIZE);
  b = create_iarray(ARRAYSIZE);
  s = create_iarray(ARRAYSIZE);
  d = create_iarray(ARRAYSIZE);
  p = create_iarray(ARRAYSIZE);
  q = create_darray(ARRAYSIZE);

  // initialize array
  for (int i = 0; i < ARRAYSIZE; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  // perform fundamental operations
  GET_TIME(start);
  for (long t = 0; t < NUM_THREADS; t++) {
    rc = pthread_create(&threads[t], NULL, fundamental_operations, (void *)t);
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
      }
  }
  GET_TIME(finish);

  printf("Sum is %d,\n", s[ARRAYSIZE - 1]);
  printf("Difference is %d,\n", d[ARRAYSIZE - 1]);
  printf("Product is %d,\n", p[ARRAYSIZE - 1]);
  printf("Quotient is %.1f,\n", q[ARRAYSIZE - 1]);
  printf("Elapsed time = %f milliseconds.\n", (finish - start) * 1000);

  pthread_exit(NULL);
  return 0;
}

void *fundamental_operations(void *threadid) {
  long tid;
  tid = (long)threadid;
  switch(tid) {
  case 0 :
    addition(s, a, b);
    break;
  case 1 :
    subtraction(d, a, b);
    break;
  case 2 :
    multiplication(p, a, b);
    break;
  case 3 :
    division(q, a, b);
    break;
  default:
    printf("Error");
  }
  pthread_exit(NULL);
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
