// ***************************************************************************
// Assignmen #5 In Class Homework
// Name: Yujin Yoshimura
// Parallel Programming Date: November 14, 2019
// Version: 1.0
// Instructor: Dr. Eduardo Colmenares
// ***************************************************************************
// This parallel program computes local and global sum of two arrays.
//
// TACC command to compile:
// mpicc YoshimuraA5Mutex.c -pthread -o YoshimuraA5Mutex_Exe
// ***************************************************************************

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 8

typedef struct thread_data thread_data;

void *getsum(void *threadid);
int* create_array(int size);

struct thread_data {
  int thread_id;
  int thread_start;
  int thread_finish;
};

const int ARRAYSIZE = 800;

int* a;
int* b;
int global_sum;
int local_sum[NUM_THREADS];
pthread_mutex_t mutex_sum;
thread_data thread_data_array[NUM_THREADS];

int main(int argc, char **argv) {
  pthread_t threads[NUM_THREADS];
  int rc;
  void* status;
  pthread_attr_t attr;
  global_sum = 0;
  a = create_array(ARRAYSIZE);
  b = create_array(ARRAYSIZE);

  // initialize array
  for (int n = 0; n < ARRAYSIZE; n++) {
    a[n] = n;
    b[n] = 2 * n;
  }

  // initialize mutex
  pthread_mutex_init(&mutex_sum, NULL);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  printf("======================================================================\n");

  // thread operations
  for (int t = 0; t < NUM_THREADS; t++) {
    // initialize thread_data_array
    thread_data_array[t].thread_id = t;
    thread_data_array[t].thread_start = t * 100;
    thread_data_array[t].thread_finish = (t + 1) * 100 - 1;
    rc = pthread_create(&threads[t], &attr, getsum, (void *)&thread_data_array[t]);
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }
  pthread_attr_destroy(&attr);

  // wait for all threads to finish operations
  for (int t = 0; t < NUM_THREADS; t++) {
    pthread_join(threads[t], &status);
  }

//  for (int t = 0; t < NUM_THREADS; t++) {
//    printf("Pthreads %d - did %3d to %3d | Local Sum = %6d | Global Sum = %6d\n", thread_data_array[t].thread_id, thread_data_array[t].thread_start, thread_data_array[t].thread_finish, local_sum[t], global_sum);
//  }
  printf("Final global sum is: %d\n", global_sum);
  printf("======================================================================\n");

  pthread_exit(NULL);
  return 0;
}

// ***************************************************************************
// Function Name: getsum
// Parameters: void*
// Return: void*
// Description: Gets local and global sum of a portion of an array.
// ***************************************************************************
void *getsum(void *threadarg) {
  int tid, start, finish, sum = 0;
  thread_data *local_data;

  local_data = (thread_data *) threadarg;
  tid = local_data->thread_id;
  start = local_data->thread_start;
  finish = local_data->thread_finish;
  for (int i = start; i <= finish; i++) {
    sum += a[i] + b[i];
  }
  pthread_mutex_lock (&mutex_sum);
  local_sum[tid] = sum;
  global_sum += sum;
  printf("Pthreads %d - did %3d to %3d | Local Sum = %6d | Global Sum = %6d\n", tid, start, finish, local_sum[tid], global_sum);
  pthread_mutex_unlock (&mutex_sum);
  pthread_exit(NULL);
}

// ***************************************************************************
// Function Name: create_array
// Parameters: int
// Return: int*
// Description: Creates an array of int with a given size.
// ***************************************************************************
int* create_array(int size) {
  int *a = (int *) calloc(size, sizeof(int));
  return a;
}
