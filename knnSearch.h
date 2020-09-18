#ifndef KNN_H
#define KNN_H

#include "vptree.h"

#define BlockSizeQuery 32

typedef struct knnresult {
	int* nidx;
	double* ndist;
	int m;
	int k;
}knnresult;

knnresult* searchQuery(vptree* root, int k, int numberOfPoints, int d2, double* query, int numberOfQuery);


#endif 

