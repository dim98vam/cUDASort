#ifndef DEFINES_H
#define DEFINES_H

#define BlockSize 64 //define blockSize to use in shared memory allocation
#define sharedMatrixSize BlockSize+1 //define shared matrix additional size=BlockSize/32-1 in order to avoid bank conflicts
#define distanceBlockSize 128
#define matrixTransposeBlock 4
//used in matrix transpose
#define matrixTransposePoints matrixTransposeBlock


//defines used in memory bank conflict free cumulative sum
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#define CONFLICT_FREE_OFFSET_double(n) ((n) >> (LOG_NUM_BANKS-1))  //-1 because since doubles are in mind the effective positions are 16 consecutive instead of 32

//defines used in nodeCalculationKernel
#define distanceCalculationPartSize distanceBlockSize
#define partitionPartSize BlockSize

//defines used in nodeCreation Kernel and buildVp __host__ function
#define maximumNumberOfNodes 8 //defines the maximum number of concurrently build nodes - defined as a power of two for convenience
#define maximumNumberOfBlocks 64 //maximum number of blocks to be created during node creation
#define nodeMemorySize maximumNumberOfNodes*maximumNumberOfBlocks //memory neccessary for parallel node creation in blocks

//define used to avoid boiler code when cuda errorChecking
#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
  }                                                         \
}

#endif
