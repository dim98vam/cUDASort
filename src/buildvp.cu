#include "vptree.h"
#include <chrono>
#include <iostream>


//----------------------------------------------------------HOST FUNCTIONS-------------------------------------------------------------------------//

/*--------------create tree wrapper function - to be called by the host in order to create vp tree with given points and size of points------------//
---------------------------returns a structure which holds the tree tables - allocation is done within the function itself-------------------------*/

vptree* buildvp(double* points,int d1,int d2)
{
	int size = d1 * d2; //holds size of matrix of points

	
	//---------------------------------------------------------------memory allocation on host and device----------------------------------------------------//
	
	int temp;
	int sizeHolder;
	
	//--------------------allocate memory on gpu--------------------
	 
	//gpu pointers
	double** startOfSequencesPoints;      //pointers to hold matrices containing pointers to matrices' segments used during partitioning
	double** startOfSequencesPointsAux;   //size of slots equals to the maximum number of concurrent nodes created on gpu and data is transferred
	double** startOfSequencesDistances;   //from host for each call of the kernel
	double** startOfSequencesDistancesAux;
	int** startOfSequencesIndexes;
	int** startOfSequencesIndexesAux;
	
	int* sizeOfSequences; //pointer to matrix holding size of subsequences to be partitioned at each level
	int* levelNodeIndexes; //pointer to matrix holding level relative indexes of nodes
	
	double* distances;  //matrices holding data for points indexes and memory for calculated distances
	double* distancesAux;
	double* pointsDevice;
	double* pointsAux;
	int* index_Device;
	int* indexAux;
	
	int* pivotIndexMatrix;  //matrix to hold temporary indexes returned from partitioning and used in quickselect
	
	int* Global_Less;    //matrices to hold values neccessary in synchronization while partitioning
	int* Global_Greater; 
	int* blockSync_Device;
	

	int* indexTree;  //matrices to hold tree data - tree is stored in matrices as a binary tree with 2i+1/2i+2 indexing for child nodes of node i
	double* vantagePointTree;
	double* treeDistancesTable;
	uint8_t* treeLeftChild;
	uint8_t* treeRightChild;
	
	double* tempTree;

	
	//cpu pointers
	double* treePoints;  //pointers pointing to tree structures 
	double* treeDistances;
	int* treeIndexes;
	uint8_t* treeLeftChild_Host;
	uint8_t* treeRightChild_Host;

	int* indexes; //pointer to matrix holding temp indexes at the start
	
	
	//gpu memory allocation
	{
		temp = ((int)floor(log2(d1))) + 1;
		sizeHolder = 1<<(temp-1);

		
		cucheck_dev(cudaMalloc((void**)&startOfSequencesPoints, sizeHolder * sizeof(double*)));      //allocate memory for pointers pointing to subsegments
		cucheck_dev(cudaMalloc((void**)&startOfSequencesPointsAux, sizeHolder * sizeof(double*)));
		cucheck_dev(cudaMalloc((void**)&startOfSequencesDistances, sizeHolder * sizeof(double*)));
		cucheck_dev(cudaMalloc((void**)&startOfSequencesDistancesAux, sizeHolder * sizeof(double*)));
		cucheck_dev(cudaMalloc((void**)&startOfSequencesIndexes, sizeHolder * sizeof(int*)));
		cucheck_dev(cudaMalloc((void**)&startOfSequencesIndexesAux, sizeHolder * sizeof(int*)));
		

		cucheck_dev(cudaMalloc((void**)&sizeOfSequences, sizeHolder * sizeof(int)));  //allocate memory for subsequences size matrix
		cucheck_dev(cudaMalloc((void**)&levelNodeIndexes, sizeHolder * sizeof(int)));

		
		cucheck_dev(cudaMalloc((void**)&distances, d1 * sizeof(double)));     //allocate memory for storing data on gpu for partitioning
		cucheck_dev(cudaMalloc((void**)&distancesAux, d1 * sizeof(double)));
		cucheck_dev(cudaMalloc((void**)&index_Device, d1 * sizeof(int)));
		cucheck_dev(cudaMalloc((void**)&indexAux, d1 * sizeof(int)));
		cucheck_dev(cudaMalloc((void**)&pointsDevice, size * sizeof(double)));
		cucheck_dev(cudaMalloc((void**)&pointsAux, size * sizeof(double)));
		

		cucheck_dev(cudaMalloc((void**)&pivotIndexMatrix,nodeMemorySize * sizeof(int))); //allocate memory for indexes returned by quickselect

		
		cucheck_dev(cudaMalloc((void**)&Global_Less, nodeMemorySize * sizeof(int)));  //allocate memory for block sync variable matrices
		cucheck_dev(cudaMalloc((void**)&Global_Greater, nodeMemorySize * sizeof(int)));
		cucheck_dev(cudaMalloc((void**)&blockSync_Device, nodeMemorySize * sizeof(int)));

		
		sizeHolder = (1<<temp) - 1;
		
		
		cucheck_dev(cudaMalloc((void**)&indexTree, sizeHolder * sizeof(int)));              //allocate memory for tree storage on gpu
		cucheck_dev(cudaMalloc((void**)&vantagePointTree, sizeHolder * d2 * sizeof(double)));
		cucheck_dev(cudaMalloc((void**)&treeLeftChild, sizeHolder * sizeof(uint8_t)));
		cucheck_dev(cudaMalloc((void**)&treeRightChild, sizeHolder * sizeof(uint8_t)));
		cucheck_dev(cudaMalloc((void**)&treeDistancesTable, sizeHolder * sizeof(double)));
		

		cucheck_dev(cudaMalloc((void**)&tempTree, sizeHolder * d2 * sizeof(double))); //allocate for temp tree


	}

	//---------------------------------allocate memory on cpu------------------------------//
	
	//allocate memory on cpu 
	{
		indexes = (int*)malloc(d1 * sizeof(int)); //allocate for temp indexes


		//allocate for tree
		treePoints= (double*)malloc(sizeHolder * d2 * sizeof(double));
		treeDistances = (double*)malloc(sizeHolder * sizeof(double));
		treeIndexes = (int*)malloc(sizeHolder * sizeof(int));
		treeLeftChild_Host = (uint8_t*)malloc(sizeHolder * sizeof(uint8_t));
		treeRightChild_Host = (uint8_t*)malloc(sizeHolder * sizeof(uint8_t));

		
		
	}

	//---------------------------------------------------------------preparation phase-------------------------------------------------------------------

	int blockCount;
	//populating index matrix and transposing points matrix 
	{
		for (int i = 0; i < d1; ++i)
		    indexes[i] = i;

	    //copyIndexMatrix
	    cudaMemcpy(index_Device, indexes, d1 * sizeof(int), cudaMemcpyHostToDevice);

	    //transpose points matrix on gpu for optimal access and free points buffer to save memory
	    cudaMemcpy(pointsAux, points,size*sizeof(double), cudaMemcpyHostToDevice);

	
		blockCount = ((d1 + matrixTransposePoints - 1) / (matrixTransposePoints)); //# of blocks used in transpose kernel - BlockSize used because of multidimensionality and shared memory size
	    matrixTranspose << <blockCount, matrixTransposeBlock, matrixTransposeBlock* d2 * sizeof(double) >> > (pointsAux, pointsDevice, d1, d2, matrixTransposePoints, matrixTransposeBlock);

	}

	//copy starting pointers and sizes for sequences to device
	{
		int sizeOfSequences_Host[] = { d1 };                           //assign starting pointers for first node
		double* startOfSequencesPoints_Host[] = { pointsDevice };
		double* startOfSequencesPointsAux_Host[] = { pointsAux };
		double* startOfSequencesDistances_Host[] = { distances };
		double* startOfSequencesDistancesAux_Host[] = { distancesAux };
		int* startOfSequencesIndexes_Host[] = { index_Device };
		int* startOfSequencesIndexesAux_Host[] = { indexAux };
		
		int levelNodeIndex_Host[] = { 0 }; //starting index is 0;


		//copy elements to device
		cudaMemcpy(startOfSequencesPoints, startOfSequencesPoints_Host, sizeof(double*), cudaMemcpyHostToDevice);             //copying to segment info holders
		cudaMemcpy(startOfSequencesPointsAux, startOfSequencesPointsAux_Host, sizeof(double*), cudaMemcpyHostToDevice);
		cudaMemcpy(startOfSequencesDistances, startOfSequencesDistances_Host, sizeof(double*), cudaMemcpyHostToDevice);
		cudaMemcpy(startOfSequencesDistancesAux, startOfSequencesDistancesAux_Host,  sizeof(double*), cudaMemcpyHostToDevice);
		cudaMemcpy(startOfSequencesIndexes, startOfSequencesIndexes_Host, sizeof(int*), cudaMemcpyHostToDevice);
		cudaMemcpy(startOfSequencesIndexesAux, startOfSequencesIndexesAux_Host,  sizeof(int*), cudaMemcpyHostToDevice);
		cudaMemcpy(sizeOfSequences, sizeOfSequences_Host,  sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(levelNodeIndexes, levelNodeIndex_Host, sizeof(int), cudaMemcpyHostToDevice); //copying starting index for level 0

	}


	//--------------------------------------------------------------tree creation phase-----------------------------------------------------------------

	//for loop for tree levels covered totally bu maximumNumberOfNodes
	size_t sharedMemSize; //variables to assist in for loop execution
	int nodesNumber;

	int offset = 0; //offset used to access matrix tables on device

	auto start =std::chrono::high_resolution_clock::now(); //start timing algorithm
	
	//first for loop
	int lastLevel = (int)floor(log2(d1));
	uint8_t isLastLevel=0;

	
	for (int i = 0; i <= (int)log2(maximumNumberOfNodes) && i <= lastLevel;++i) 
	{
		nodesNumber = 1<<i; //number of nodes in current level
		sharedMemSize = 4 * nodesNumber * sizeof(int) + 4 * nodesNumber * sizeof(int*) + 8 * nodesNumber * sizeof(double*); //size required for kernel shared memory

		if (i == lastLevel)
			isLastLevel = 1;
		
		//call the kernel
		nodeCreation << <1, nodesNumber, sharedMemSize >> > (startOfSequencesPoints, startOfSequencesPointsAux, startOfSequencesDistances, startOfSequencesDistancesAux,
			                                                    startOfSequencesIndexes, startOfSequencesIndexesAux, sizeOfSequences, levelNodeIndexes, d2, d1, Global_Less, 
			                                                    Global_Greater, blockSync_Device,pivotIndexMatrix, indexTree+offset, treeDistancesTable+offset, vantagePointTree+offset, 
			                                                    sizeHolder, treeLeftChild+offset, treeRightChild+offset, i,isLastLevel);

		offset += nodesNumber; //update offset for next iteration	
		
		cudaDeviceSynchronize(); //wait for level creation completion
	}


	//second for loop
	int secondaryOffset; //used to offset buffer arrays
	int numberOfBlocks;
	int numberOfIterations;
	int gridSize;
	
	for (int i=(int)log2(maximumNumberOfNodes)+1;i<= lastLevel;++i)
	{
		nodesNumber = 1<<i; //number of nodes in current level
		numberOfBlocks = nodesNumber / maximumNumberOfNodes;  //number of blocks needed to create level
		
		if (numberOfBlocks > maximumNumberOfBlocks) {
			numberOfIterations = numberOfBlocks / maximumNumberOfBlocks; //passes needed to create all nodes in current level
			gridSize = maximumNumberOfBlocks;
		}
		else {
			numberOfIterations = 1; //if maximum number of blocks covers current level then simply assign less blocks
			gridSize = numberOfBlocks;
		}

		if (i == lastLevel)
			isLastLevel = 1;

		secondaryOffset = 0; //initialize secondary offset for inner loop

		//for loop to call all kernels for level
		for(int y=0;y<numberOfIterations;++y)
		{
			nodeCreation << <gridSize, maximumNumberOfNodes, sharedMemSize >> > (startOfSequencesPoints + secondaryOffset, startOfSequencesPointsAux + secondaryOffset, startOfSequencesDistances + secondaryOffset,
				                                                             startOfSequencesDistancesAux + secondaryOffset, startOfSequencesIndexes + secondaryOffset, startOfSequencesIndexesAux + secondaryOffset,
				                                                             sizeOfSequences + secondaryOffset, levelNodeIndexes + secondaryOffset, d2, d1, Global_Less, Global_Greater, blockSync_Device, pivotIndexMatrix,
				                                                             indexTree + offset, treeDistancesTable + offset, vantagePointTree + offset, sizeHolder, treeLeftChild + offset,
				                                                             treeRightChild + offset, i,isLastLevel);
		  
			secondaryOffset += gridSize*maximumNumberOfNodes; //update secondary offset for buffer dereferencing
			cudaDeviceSynchronize(); //wait for level creation completion
		}

		offset += nodesNumber; //update treeIndex offset

	}

	
	auto stop = std::chrono::high_resolution_clock::now(); //record end of algorithm
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	printf("Tree creation took: "); 
	
	std::cout << duration.count() ;

	printf(" ms\n\n");

	//-------------------------------------------------------------------copying data back to host memory-------------------------------------------------------
	{
		cudaMemcpy(treeDistances, treeDistancesTable, sizeHolder * sizeof(double), cudaMemcpyDeviceToHost);  //copy elements that dont require transposing
		cudaMemcpy(treeIndexes, indexTree, sizeHolder * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(treeLeftChild_Host, treeLeftChild, sizeHolder * sizeof(uint8_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(treeRightChild_Host, treeRightChild, sizeHolder * sizeof(uint8_t), cudaMemcpyDeviceToHost);

		
		
		blockCount = ((sizeHolder + matrixTransposePoints -1) / (matrixTransposePoints));
		matrixTransposeToHost << <blockCount, matrixTransposeBlock, matrixTransposePoints*d2*sizeof(double) >> > (vantagePointTree,tempTree,d2, sizeHolder,matrixTransposePoints);
		
	
		cudaMemcpy(treePoints, tempTree, sizeHolder* d2 * sizeof(double), cudaMemcpyDeviceToHost);

		
		//free table pointers
		cudaFree(tempTree);
		cudaFree(vantagePointTree);
		cudaFree(treeDistancesTable);
		cudaFree(indexTree);
		cudaFree(treeLeftChild);
		cudaFree(treeRightChild);

	}

	//----------------------------------------------------------cleaning up gpu memory--------------------------------------------------------------------------
	{
		cudaFree(startOfSequencesPoints);
		cudaFree(startOfSequencesPointsAux);
		cudaFree(startOfSequencesDistances);
		cudaFree(startOfSequencesDistancesAux);
		cudaFree(startOfSequencesIndexes);
		cudaFree(startOfSequencesIndexesAux);
		
		cudaFree(sizeOfSequences);
		cudaFree(levelNodeIndexes);
		
		cudaFree(distances);
		cudaFree(distancesAux);
		cudaFree(pointsDevice);
		cudaFree(pointsAux);
		cudaFree(index_Device);
		cudaFree(indexAux);

		cudaFree(pivotIndexMatrix);
		cudaFree(Global_Less);
		cudaFree(Global_Greater);
		cudaFree(blockSync_Device);
	}
	
	
	//----------------------------------------------------------------------------return root-------------------------------------------------------------------
	vptree* root = (vptree*)malloc(sizeof(vptree)); //allocate for root of created tree

	root->points = treePoints;
	root->leftChild = treeLeftChild_Host;
	root->rightChild = treeRightChild_Host;
	root->indexes = treeIndexes;
	root->md = treeDistances;

	root->runningIndex = 0;
	root->d2 = d2;

	return root;
}


//-----------------------------------------------------utility host functions----------------------------------------------------------------------//
vptree* getInner(vptree* node)
{
	int treeIdx = node->runningIndex;

	if (((node->leftChild)[treeIdx]) & 1) //if left child exists
	{
		vptree* child = (vptree*)malloc(sizeof(vptree)); //allocate for root of created tree

		child->points = node->points;
		child->leftChild = node->leftChild;
		child->rightChild = node->rightChild;
		child->indexes = node->indexes;
		child->md = node->md;
		child->d2 = node->d2;

		child->runningIndex = 2*treeIdx+1;

		return child;
	}

	return NULL;

}


vptree* getOuter(vptree* node)
{
	int treeIdx = node->runningIndex;

	if (((node->rightChild)[treeIdx]) & 1) //if left child exists
	{
		vptree* child = (vptree*)malloc(sizeof(vptree)); //allocate for root of created tree

		child->points = node->points;
		child->leftChild = node->leftChild;
		child->rightChild = node->rightChild;
		child->indexes = node->indexes;
		child->md = node->md;
		child->d2 = node->d2;

		child->runningIndex = 2*treeIdx+2;

		return child;
	}

	return NULL;
}


double getMD(vptree* node)
{
	
	int treeIdx = node->runningIndex;
	
	return (node->md)[treeIdx];
}

double* getVP(vptree* node)
{
	int treeIdx = (node->runningIndex) * (node->d2);
	return (node->points) + treeIdx;
}

int getIDX(vptree* node)
{
	int treeIdx = node->runningIndex;
	return (node->indexes)[treeIdx];
}

