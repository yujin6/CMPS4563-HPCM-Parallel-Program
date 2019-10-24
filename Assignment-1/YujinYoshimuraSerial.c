/**
* @author: Yujin Yoshimura
* @created: Oct. 15, 2019
* @version: 1.0
* @semester: Fall 2019
* @course: CMPS4563
* @instructor: Dr. Eduardo Colmenares
*
* This sequential program visits 640000 locations of an integer array, and
* Computes the sum of all integers in this array.
*
* TACC command to compile:
* mpicc YujinYoshimuraSerial.c -o YujinYoshimuraSerial_Exe
*/

#include <stdio.h>
#include "timer.h"

const long int ARRAY_SIZE = 640000;

int main(int argc, char **argv)
{
  long int my_array[ARRAY_SIZE];
  long int sum = 0;
  double start, finish;

  // Initialize the array
  for(long int i = 0; i < ARRAY_SIZE; i++){
    my_array[i] = i + 1;
  }

  GET_TIME(start);

  // Add each elements in the array
  for(long int j = 0; j < ARRAY_SIZE; j++){
    sum += my_array[j];
  }
  GET_TIME(finish);

  printf("\nThe summation of all numbers is: %ld\n", sum);
  printf("Elapsed time = %e seconds.\n", finish - start);

  return 0;
}
