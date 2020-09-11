#ifndef UTILITIES_H
#define UTILITIES_H

#include "vptree.h"

__global__
void  matrixTranspose(double* oldMatrix, double* newMatrix, int d1, int d2, int blockPoints, int chunk);

__global__
void matrixTransposeToHost(double* oldMatrix, double* newMatrix, int d1, int d2, int blockPoints);

__global__
void copyKernel(double* pointsMatrixToCopy, double* pointsMatrixDest, int* indexMatrixToCopy, int* indexMatrixDest, int size, int d1, int d2, int partSize);


#endif