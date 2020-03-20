// ***************************************************************************
// Assignment #3
// Name: Yujin Yoshimura
// Parallel Programming Date: March 12, 2020
// ***************************************************************************
// References:
//
// NVIDIA CUDA Toolkit Documentation
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
//
// NVIDIA Developer Blog
// https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
// https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/
//
// Stack Overflow
// https://stackoverflow.com/questions/22520209/programmatically-retrieve-maximum-number-of-blocks-per-multiprocessor
// ***************************************************************************
// This sequential program queries device properties.
//
// For Turing, use the script on the same directory to compile and run.
//
// For TACC Maverick 2, use the script to run this code:
// sbatch YujinYoshimura.script
//
// Otherwise, use TACC Maverick 2 command to compile:
// nvcc YujinYoshimura.cu -o YujinYoshimura_Exe
// TACC Maverick 2 command to run executable:
// ./YujinYoshimura_Exe
// ***************************************************************************

#include <cuda.h>
#include <stdio.h>

// ***************************************************************************
// Function Name: main
// Parameters: int, char**
// Return: int
// Description: Main function of the program.
// ***************************************************************************
int main(int argc, char **argv) {
  int nDevices, blocks, version;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cudaDeviceGetAttribute(&version, cudaDevAttrComputeCapabilityMajor, i);
    // According to Feature Support per Compute Capability
    if (version < 3) {
      blocks = 8;
    } else if (version < 5) {
      blocks = 16;
    } else {
      blocks = 32;
    }
    printf("=============================================================\n");
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Size of shared memory per block: %zu\n", prop.sharedMemPerBlock);
    printf("  Number of registers per block: %d\n", prop.regsPerBlock);
    printf("  The corresponding warp size: %d\n", prop.warpSize);
    printf("  The maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  The maximum number of threads that we can have\n");
    printf("    for a 3D layout: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  The maximum grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max number of blocks per streaming multiprocessor: %d \n", blocks);
  }
  printf("=============================================================\n");

 	return EXIT_SUCCESS;
}
