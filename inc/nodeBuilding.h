#ifndef NODE_H
#define NODE_H

#include "vptree.h"
#include <stdint.h>

__global__
void nodeCreation(double** startOfSequencesPoints, double** startOfSequencesPointsAux, double** startOfSequencesDistances, double** startOfSequencesDistAux,
	int** startOfSequencesIndexes, int** startOfSequencesIndAux, int* sizeOfSequences, int* nodeLevelndex, int d2, int d1, int* GlobalLess, int* GlobalGreater,
	int* blockSync, int* pivotIndexMatrix, int* treeIndexTable, double* treeDistancesTable, double* treePointsTable, int treeMatrixSize, uint8_t* LeftChild,
	uint8_t* RightChild, int treeLevel, uint8_t isLastLevel);

#endif
