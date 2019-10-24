/**
* @author: Yujin Yoshimura
* @created: Oct. 15, 2019
* @version: 1.0
* @semester: Fall 2019
* @course: CMPS4563
* @instructor: Dr. Eduardo Colmenares
*
* This parallel program visits 640000 locations of an integer array, and
* Computes the sum of all integers in this array.
*
* TACC command to compile:
* mpicc YujinYoshimuraParallelV1.c -o YujinYoshimuraParallelV1_Exe
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const long int ARRAY_SIZE = 640000;

int main(int argc, char **argv)
{
  int comm_sz;
  int my_rank;
  long int* my_array;
  long int local_array_size;
  long int* local_array;
  long int local_sum = 0;
  long int final_sum = 0;
  long int i;
  double start, finish;

  MPI_Init(NULL, NULL); 
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
  local_array_size = ARRAY_SIZE / comm_sz;
  local_array = (long int*) calloc(local_array_size, sizeof(long int));

  if (my_rank == 0) {
    // Process 0 is the only process that owns the array
    my_array = (long int*) calloc(ARRAY_SIZE, sizeof(long int));

    // Initialize the array
    for (i = 0; i < ARRAY_SIZE; i++) {
      my_array[i] = i + 1;
    }
  }

  start = MPI_Wtime();

  // Distributes the values to each process
  MPI_Scatter(my_array, local_array_size, MPI_LONG, local_array, local_array_size, MPI_LONG, 0, MPI_COMM_WORLD);

  for (i = 0; i < local_array_size; i++) {
    local_sum += local_array[i];
  }

  // Collects sums from each process
  MPI_Reduce(&local_sum, &final_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD); 

  finish = MPI_Wtime();

  if (my_rank == 0) {
    printf("\nThe summation of all numbers is: %ld\n", final_sum);
    printf("Elapsed time = %e seconds.\n", finish - start);
  }

  MPI_Finalize(); 

  return 0;
}
