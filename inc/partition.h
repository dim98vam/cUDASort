#ifndef PARTITION_H
#define PARTITION_H

#include "vptree.h"

__global__ void
partitionKernel(double* DataSet_Device, double* auxBuffer_Device, double* pIvot, int* pivotIndex, int size, int partSize,
	int* Global_Less, int* Global_Greater, int* blockSync, int* index_Device, int* indexAux, double* pointsDevice,
	double* pointsAux, int d1, int d2, int pivotPlace);

__global__ void
distanceCalculation(double* pointMatrix, double* distanceMatrix, int numOfpoints, int dimensions, double* pivotPoint,
	int partSize, int totalPoints);


#endif

