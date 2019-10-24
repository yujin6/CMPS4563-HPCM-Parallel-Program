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
#include <mpi.h>

const long int ARRAY_SIZE = 640000;

int main(int argc, char **argv)
{
  int comm_sz;
  int my_rank;

  MPI_Init(NULL, NULL); 
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

  if (my_rank == 0) {
    // Process 0 is the only process that owns the array
    long int my_array[ARRAY_SIZE];
    long int summation[comm_sz];
    long int local_value, i;
    long int local_sum = 0;
    long int final_sum = 0;
    int q;
    double start, finish;

    // Initialize the array
    for (i = 0; i < ARRAY_SIZE; i++) {
      my_array[i] = i + 1;
    }

    start = MPI_Wtime();

    // Distributes the values to each process
    for (i = 0; i < ARRAY_SIZE; i++) {
      // Determines which process to be in charge of the value
      q = i % comm_sz;
      local_value = my_array[i];

      // if Process 0 is in charge, just add up
      if (q == 0) {
        local_sum += local_value;
      } else {
        MPI_Send(&local_value, 1, MPI_LONG, q, 0, MPI_COMM_WORLD); 
      }
    }

    // Collects sums from each process
    summation[0] = local_sum;
    for (q = 1; q < comm_sz; q++) {
      MPI_Recv(&local_sum, 1, MPI_LONG, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      summation[q] = local_sum;
    }

    // Process 0 takes care of final addition
    for (q = 0; q < comm_sz; q++) {
      final_sum += summation[q];
    }

    finish = MPI_Wtime();

    printf("\nThe summation of all numbers is: %ld\n", final_sum);
    printf("Elapsed time = %e seconds.\n", finish - start);

  } else {
    long int local_array_size = ARRAY_SIZE / comm_sz;
    long int local_value, i;
    long int local_sum = 0;

    
    // Add up each elements being passed from Process 0
    for (i = 0; i < local_array_size; i++) {
      MPI_Recv(&local_value, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_sum += local_value;
    }
    
    // Send the sum to Process 0
    MPI_Send(&local_sum, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD); 
  }

  MPI_Finalize(); 

  return 0;
}
