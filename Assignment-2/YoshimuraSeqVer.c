// ***************************************************************************
// Assignment #2
// Name: Yujin Yoshimura
// Parallel Programming Date: October 31, 2019
// ***************************************************************************
// This parallel program demonstrates Fast Fourier Transform,
// with Cooley-Tukey FFT Algorithm, also known as Radix-2.
//
// For Turing, use the script on the same directory to compile and run.
// TACC Stampede command to compile:
// mpicc YoshimuraSeqVer.c -o YoshimuraSeqVer_Exe
// ***************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "timer.h"

const int ITERATION = 3;

typedef struct Complex Complex;

Complex* create_array(int size);
void initialize_array(Complex* a, int size);
void initialize_sample(Complex* sample, int size);
Complex* reverse_sample(Complex* sample, int length);
int reverse_bit(int bit, int length);
Complex* shift_sample(Complex* sample, int size, int itr);
int shift_bit(int bit, int itr);
Complex negate(Complex a);
Complex add(Complex a, Complex b);
Complex multiply(Complex a, Complex b);
Complex get_factor(int id, int global_size);
void butterfly(Complex* sample, int size, int itr);
void print_sample(Complex* sample, int size);

struct Complex {
  int id;
  float re;
  float im;
};

// ***************************************************************************
// Function Name: main
// Parameters: int, char**
// Return: int
// Description: Main function of the program.
// ***************************************************************************
int main(int argc, char **argv) {
  int length = 3;
  int global_size;
  Complex *global_sample;
  double start, finish, average = 0;
  double elapsed[ITERATION];
  int i;

  // Scans the problem size from script
  sscanf(argv[1],"%d",&length);
  global_size = pow(2, length);

  // Initializes sample and rearrange order
  global_sample = create_array(global_size);
  initialize_sample(global_sample, global_size);
  global_sample = reverse_sample(global_sample, length);

  for (i = 0; i < ITERATION; i++) {
    GET_TIME(start);
    for (int itr = 0; itr < length; itr++) {
      // Performs butterfly operation
      butterfly(global_sample, global_size, itr);

      // Rearranges global sample for the next iteration
      global_sample = shift_sample(global_sample, global_size, itr);
    }

    GET_TIME(finish);
    elapsed[i] = finish - start;
    average += elapsed[i];

    // Prints result of FFT
    if (i == 0) {
      printf("Number of Processes: 1\n");
      print_sample(global_sample, global_size);
    }
    printf("Elapsed time = %e seconds.\n", elapsed[i]);
  }

  average = average / ITERATION;
  printf("Average elapsed time = %e seconds.\n", average);

  return 0;
}

// ***************************************************************************
// Function Name: create_array
// Parameters: int
// Return: Complex*
// Description: Creates an array of Complex with a given size.
// ***************************************************************************
Complex* create_array(int size) {
  Complex *a = (Complex *) calloc(size, sizeof(Complex));
  return a;
}

// ***************************************************************************
// Function Name: initialize_array
// Parameters: Complex*, int
// Return: void
// Description: Initializes an array of Complex by setting 0 for all entries.
// ***************************************************************************
void initialize_array(Complex* a, int size) {
  // Set 0 for all entries
  for (int i = 0; i < size; i++) {
    a[i].id = i;
    a[i].re = 0;
    a[i].im = 0;
  }
}

// ***************************************************************************
// Function Name: initialize_sample
// Parameters: Complex*, int
// Return: void
// Description: Initializes an array of sample with first 8 entries being
//              significant entries, and leave the rest as 0's.
// ***************************************************************************
void initialize_sample(Complex* sample, int size) {
  initialize_array(sample, size);

  // Set specific values for first 8 entries
  sample[0].re = 3.6;
  sample[0].im = 2.6;
  sample[1].re = 2.9;
  sample[1].im = 6.3;
  sample[2].re = 5.6;
  sample[2].im = 4.0;
  sample[3].re = 4.8;
  sample[3].im = 9.1;
  sample[4].re = 3.3;
  sample[4].im = 0.4;
  sample[5].re = 5.9;
  sample[5].im = 4.8;
  sample[6].re = 5.0;
  sample[6].im = 2.6;
  sample[7].re = 4.3;
  sample[7].im = 4.1;
}

// ***************************************************************************
// Function Name: reverse_sample
// Parameters: Complex*, int
// Return: Complex*
// Description: Rearranges sample order by reverse_bit order.
// ***************************************************************************
Complex* reverse_sample(Complex* sample, int length) {
  int size = pow(2, length);
  Complex *new_sample = create_array(size);
  for(int i = 0; i < size; i++) {
    new_sample[i] = sample[reverse_bit(i, length)];
  }
  return new_sample;
}

// ***************************************************************************
// Function Name: reverse_bit
// Parameters: int, int
// Return: int
// Description: Reverses binary expression of an int with the given length.
// ***************************************************************************
int reverse_bit(int bit, int length) {
  int reverse = 0;
  int mask = 1;
  int remainder;

  for(int i = 0; i < length; i++){
    reverse = (reverse << 1);
    remainder = bit & mask;
    reverse += remainder;
    bit = (bit >> 1);
  }
  return reverse;
}

// ***************************************************************************
// Function Name: shift_sample
// Parameters: Complex*, int, int
// Return: Complex*
// Description: Rearranges sample order by shift_bit order.
// ***************************************************************************
Complex* shift_sample(Complex* sample, int size, int itr) {
  Complex *new_sample = create_array(size);
  for(int i = 0; i < size; i++) {
    new_sample[i] = sample[shift_bit(i, itr)];
  }
  return new_sample;
}

// ***************************************************************************
// Function Name: shift_bit
// Parameters: int, int, int
// Return: int
// Description: Shifts binary expression of an int with the given length.
// ***************************************************************************
int shift_bit(int bit, int itr) {
  int mask = pow(2, itr + 1) - 1;
  int remainder = bit & mask;
  bit = (bit >> (itr + 1));
  int r = remainder & 1;
  remainder = (remainder >> 1) | (r << itr);
  bit = (bit << (itr + 1)) | remainder;
  return bit;
}

// ***************************************************************************
// Function Name: negate
// Parameters: Complex
// Return: Complex
// Description: Negates a complex number.
//              Complex number is negated as:
//              - (a + bi) = (-a) + (-b) i.
// ***************************************************************************
Complex negate(Complex a) {
  Complex inverse;
  inverse.id = a.id;
  inverse.re = -a.re;
  inverse.im = -a.im;
  return inverse;
}

// ***************************************************************************
// Function Name: add
// Parameters: Complex, Complex
// Return: Complex
// Description: Adds two complex numbers, and returns the sum.
//              Complex numbers are added as:
//              (a + bi) + (c + di) = (a + c) + (b + d) i.
// ***************************************************************************
Complex add(Complex a, Complex b) {
  Complex sum;
  sum.id = a.id;
  sum.re = a.re + b.re;
  sum.im = a.im + b.im;
  return sum;
}

// ***************************************************************************
// Function Name: multiply
// Parameters: Complex, Complex
// Return: Complex
// Description: Multiplies two complex numbers, and returns the product.
//              Complex numbers are multipled as:
//              (a + bi) * (c + di) = (ac - bd) + (ad + bc) i.
// ***************************************************************************
Complex multiply(Complex a, Complex b) {
  Complex product;
  product.id = a.id;
  product.re = a.re * b.re - a.im * b.im;
  product.im = a.re * b.im + a.im * b.re;
  return product;
}

// ***************************************************************************
// Function Name: get_factor
// Parameters: int, int
// Return: Complex
// Description: Calculates the factor for each sample to be multiplied with
//              for the purpose of Fast Fourier Transform.
//              for nth sample out of size N, the factor is calculated as:
//              cos (2 * pi * n/N) - i sin (2 * pi * n/N).
// ***************************************************************************
Complex get_factor(int id, int swap_size) {
  Complex factor;
  factor.id = id;
  factor.re = cos(2 * M_PI * (float)id / (float)swap_size);
  factor.im = -sin(2 * M_PI * (float)id / (float)swap_size);
  return factor;
}

// ***************************************************************************
// Function Name: butterfly
// Parameters: Complex*, int, int
// Return: void
// Description: Performs butterfly operation according to the Fast Fourier
//              Transform Algorithm. The butterfly operation is calculated as:
//              X[k] = E[k] + e ^ (- 2 * pi * i * k / N) * O[k]
//              X[k + N/2] = E[k] - e ^ (- 2 * pi * i * k / N) * O[k]
//              For E[k] to be an even element, and O[k] to be an odd element.
//              The e ^ (- 2 * pi * i * k / N) part is a factor omega.
// ***************************************************************************
void butterfly(Complex* sample, int size, int itr) {
  int swap_size = pow(2, itr + 1);
  Complex *w = create_array(swap_size / 2);
  Complex even, odd;

  // Create an array of factors omega
  for (int i = 0; i < swap_size / 2; i++) {
    w[i] = get_factor(i, swap_size);
  }

  // Perform butterfly operation in new sample
  for (int i = 0; i < size / 2; i++) {
    // For even
    even = add(sample[2 * i], multiply(sample[2 * i + 1], w[i % (swap_size / 2)]));
    // For odd
    odd = add(sample[2 * i], multiply(sample[2 * i + 1], negate(w[i % (swap_size / 2)])));
    odd.id = sample[2 * i + 1].id;
    // Put them back to the sample
    sample[2 * i] = even;
    sample[2 * i + 1] = odd;
  }
}

// ***************************************************************************
// Function Name: print_sample
// Parameters: Complex*, int
// Return: void
// Description: Prints the sample data in an organized format.
// ***************************************************************************
void print_sample(Complex* sample, int size) {
  float significance = 0;
  int i = 0;
  printf("TOTAL PROCESSED SAMPLES: %d\n", size);
  printf("=================================================\n");
  do {
    printf("XR[%5d]: %8.4f       XI[%5d]: %8.4f\n", sample[i].id, sample[i].re, sample[i].id, sample[i].im);
    significance = sample[i].re;
    i++;
    } while (significance != 0);
  printf("=================================================\n");
  printf("XR[%5d]: %8.4f       XI[%5d]: %8.4f\n", sample[i].id, sample[i].re, sample[i].id, sample[i].im);
  printf("     :                          :\n");
  printf("     :                          :\n");
  printf("XR[%5d]: %8.4f       XI[%5d]: %8.4f\n", sample[size-1].id, sample[size-1].re, sample[size-1].id, sample[size-1].im);
  printf("=================================================\n");
}
