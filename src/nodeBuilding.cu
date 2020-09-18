#include "vptree.h"

/*kernel used to create nodes in tree - called by cpu at each level of the tree and itself(using dynamic parallelism) calls distance-//
--------------------------------calculation and partition kernels to manipulate each nodes data--------------------------------------
startOfSequencesPoints matrix contains pointers to the start of each subsequence of points corresponding to a node - sizeOfSequences-//
holds the size for each such sebsequence - startOfSequencesDistances contains the preallocated space for distance calculation--------//
----------------------------------streams used are taken from preallocated streams matrix--------------------------------------------*/

__global__ void 
__launch_bounds__(16,16)
nodeCreation(double** startOfSequencesPoints, double** startOfSequencesPointsAux, double** startOfSequencesDistances, double** startOfSequencesDistAux,
	int** startOfSequencesIndexes, int** startOfSequencesIndAux, int* sizeOfSequences, int* nodeLevelndex, int d2, int d1, int* GlobalLess, int* GlobalGreater,
	int* blockSync, int* pivotIndexMatrix, int* treeIndexTable, double* treeDistancesTable, double* treePointsTable, int treeMatrixSize, uint8_t* LeftChild,
	uint8_t* RightChild, int treeLevel, uint8_t isLastLevel)
{

	//assigning shared memory for later global memory accessing
	extern __shared__ int sharedMatrixBuffer[]; //sharedMatrix to hold segment sizes for next tree level - used to allow for coalesced global memory accesses
												//when writing sizes for next tree level - same for double* and int* types - bank conflicts not dealt with here

	int* sharedSizes = sharedMatrixBuffer;

	int** sharedInd = (int**)(sharedSizes + 2 * blockDim.x);
	int** sharedIndAux = (sharedInd + 2 * blockDim.x);

	double** sharedSequencesPoints = (double**)(sharedIndAux + 2 * blockDim.x);
	double** sharedSequencesPointsAux = (sharedSequencesPoints + 2 * blockDim.x);

	double** sharedSequencesDistances = (sharedSequencesPointsAux + 2 * blockDim.x);
	double** sharedSequencesDistancesAux = (sharedSequencesDistances + 2 * blockDim.x);

	int* sharedLevelIndex = (int*)(sharedSequencesDistancesAux + 2 * blockDim.x);

	//create stream for concurrent execution
	cudaStream_t streams;
	cudaStreamCreateWithFlags(&streams, cudaStreamNonBlocking);



	//---------------------------------------------------------node creation part------------------------------------------------

	int tid = threadIdx.x;
	int offset= blockIdx.x * blockDim.x + tid;
	int size = sizeOfSequences[offset]; //get size for current sequence
	int sizeHolder = size; //used to avoid two accesses in global memory
	int nodeIndex = nodeLevelndex[offset]; //used to hold # of node in level since storing breaks the order in favor of memory space and coalesced accesses


	if (size != 0) //if there are elements to process
	{
		treeIndexTable[nodeIndex] = *(*(startOfSequencesIndexes + offset) + size - 1); //add vantage point index - vantage point is taken as the last in 
																		  //current subsequence

		for (int i = 0; i < d2; ++i)
		{
			treePointsTable[nodeIndex + i * treeMatrixSize] = *(*(startOfSequencesPoints + offset) + size - 1 + i * d1); //write vantage point into tree table - accesses are not coalesced
																										//when reading global memory but no other method was found or justified
																										//as efficient
		}

		if (size > 1) //if thread has a node to process in current level proceed with required actions
		{
			--size; //account for vantage point in table

			//call distance calculation kernel for assigned part of points with vp being the last element 
			int blockCount = ((size + distanceCalculationPartSize - 1) / (distanceCalculationPartSize)); //calculate number of blocks to call

			//initializing pointers for current node
			
			double* points = startOfSequencesPoints[offset]; //pointers to tableOfPoints - account for block's part
			double* pointsAux = startOfSequencesPointsAux[offset];

			double* distances = startOfSequencesDistances[offset]; //pointers to tableOfDistances
			double* distancesAux = startOfSequencesDistAux[offset];

			int* indexes = startOfSequencesIndexes[offset]; //pointers to indexes
			int* indexesAux = startOfSequencesIndAux[offset];


			//calculate distnaces from vantage point
			distanceCalculation << <blockCount, distanceBlockSize, d2 * sizeof(double), streams >> > (points, distances, size, d2, points + size, distanceCalculationPartSize, d1);
			

			//implement a quickselect algorithm to divide the sequence into to halves						 

			int piVot; //assign expected pivot position for block subsequence
			if ((size + 1) & 1)
				piVot = size / 2; //odd # of points
			else
				piVot = size / 2 + 1; //even # of points


			int tempPivot; //variables helping in quickselect implementation
			double* tempPointer;
			int* tempPointerInt;

			int counter = 0;


			//quickselect implementation
			while (1) //quickselect part 
			{
				if (size == 1) //if after accounting for vantage point only one element remains simply add the distance as node's median distance
				{

					if (counter == 0) //account for tree with only two elements and do the copy to aux
					{
						startOfSequencesIndAux[offset] = startOfSequencesIndexes[offset]; //do a dummy swap to have the pointers right in the end
						startOfSequencesPointsAux[offset] = startOfSequencesPoints[offset];

						cudaDeviceSynchronize(); //in case only two elements exist call cudaDeviceSynchronize for distances to be calculated
					}

					treeDistancesTable[nodeIndex] = *distances;

					break;
				}

				blockCount = (int)((size + partitionPartSize - 1) / (partitionPartSize)); //calculate new blockCount for subsequence in partition
				
				*(GlobalLess + offset) = 0; //initialize global less/greater and blockSync memory positions - accesses are coalesced for speed
				*(GlobalGreater + offset) = size;
				*(blockSync + offset) = blockCount;


				//call partition kernel for current subsequence - pivot point is chosen as the last point in distance matrix
				partitionKernel << <blockCount, BlockSize, 0, streams >> > (distances, distancesAux, distances + size - 1, pivotIndexMatrix + offset, size, partitionPartSize,
					GlobalLess + offset, GlobalGreater + offset, blockSync + offset, indexes, indexesAux, points, pointsAux, d1, d2, size);


				if (counter & 1) //if counter is odd then copy aux matrix to main matrix to maintain all points partitioned in one matrix for next level
				{
					copyKernel << <blockCount, BlockSize, 0, streams >> > (pointsAux, points, indexesAux, indexes, size, d1, d2, partitionPartSize);
				}

				cudaDeviceSynchronize(); //wait for partition and copy kernels to complete

				tempPivot = *(pivotIndexMatrix + offset); //get pivot position returned from partition kernel

				if (tempPivot == piVot) //if pivot is in its correct position
				{
					treeDistancesTable[nodeIndex] = distancesAux[piVot - 1]; //-1 because piVot is one based
					break;
				}

				//update size/blockCount parameters for next pass
				if (tempPivot > piVot) //if tempPivot is to the right of the requested pivot index
					size = tempPivot - 1; //exclude tempPivot from new subsequence - piVot value remains unchanged as well as starting pointers' index 
				else
				{
					size -= tempPivot; //only include elements to the right of tempPivot

					//update pointers
					distances += tempPivot;    //distances part
					distancesAux += tempPivot;

					points += tempPivot;       //points part
					pointsAux += tempPivot;

					indexes += tempPivot;     //indexes part
					indexesAux += tempPivot;

					//update piVot
					piVot -= tempPivot;
				}


				tempPointer = distances; //swap distance matrix pointers for next pass
				distances = distancesAux;
				distancesAux = tempPointer;

				tempPointer = points; //points swap part
				points = pointsAux;
				pointsAux = tempPointer;

				tempPointerInt = indexes;  //indexes swap part
				indexes = indexesAux;
				indexesAux = tempPointerInt;

				//update counter
				++counter;

			}

		}
		else  //if this is a leaf
		{
			treeDistancesTable[nodeIndex] = 0.0;  //no median distance exists
		}

		//assign sizes and indexes for nodes in next tree level - shared memory used to result in coalesced memory accesses in global memory
		//pointers to be used in next tree level are those pointing in current auxiliary memory

		{
			int tempSizeLeft;             //used to avoid unnecessary global accesses and calculations
			int tempSizeRight;
			void* pointerHolder;

			tempSizeRight = (sizeHolder - 1) / 2; //holds size for right child segment

			//determine sife of left child segment
			if (sizeHolder & 1) //if size is odd #
			{
				tempSizeLeft = tempSizeRight;
			}
			else                //if size is even #
			{
				tempSizeLeft = (sizeHolder - 1) / 2 + 1;
			}


			//assigning possible child indexes
			{
				if (tempSizeLeft != 0) //if left subsequence exists then signal left child for current tree node exists
					LeftChild[nodeIndex] = 1;
				else
					LeftChild[nodeIndex] = 0;

				if (tempSizeRight != 0) //same for right child 
					RightChild[nodeIndex] = 1;
				else
					RightChild[nodeIndex] = 0;
			}




			if (!isLastLevel) //if this isnt the last level do the assignments
			{ //make assignments
				sharedSizes[tid * 2] = tempSizeLeft; //assign sizes for child nodes 
				sharedSizes[tid * 2 + 1] = tempSizeRight;

				pointerHolder = (void*)startOfSequencesPointsAux[offset];  //assign point and auxPoints pointers for child nodes
				sharedSequencesPoints[2 * tid] = (double*)pointerHolder;
				sharedSequencesPoints[2 * tid + 1] = (double*)pointerHolder + tempSizeLeft;

				pointerHolder = startOfSequencesPoints[offset];
				sharedSequencesPointsAux[2 * tid] = (double*)pointerHolder;
				sharedSequencesPointsAux[2 * tid + 1] = (double*)pointerHolder + tempSizeLeft;

				pointerHolder = startOfSequencesIndAux[offset]; //assign indexes and indexesAux pointers for child nodes
				sharedInd[tid * 2] = (int*)pointerHolder;
				sharedInd[tid * 2 + 1] = (int*)pointerHolder + tempSizeLeft;

				pointerHolder = startOfSequencesIndexes[offset];
				sharedIndAux[tid * 2] = (int*)pointerHolder;
				sharedIndAux[tid * 2 + 1] = (int*)pointerHolder + tempSizeLeft;

				pointerHolder = startOfSequencesDistances[offset];  //assign distances and DistAux pointers for child nodes
				sharedSequencesDistances[tid * 2] = (double*)pointerHolder;
				sharedSequencesDistances[tid * 2 + 1] = (double*)pointerHolder + tempSizeLeft;

				pointerHolder = startOfSequencesDistAux[offset];
				sharedSequencesDistancesAux[2 * tid] = (double*)pointerHolder;
				sharedSequencesDistancesAux[2 * tid + 1] = (double*)pointerHolder + tempSizeLeft;

				sharedLevelIndex[2 * tid] = nodeIndex * 2; //assign indexes of child nodes in next  tree level 
				sharedLevelIndex[2 * tid + 1] = nodeIndex * 2 + 1;
			}
		}

	}
	else if (!isLastLevel) //if no elements exist simply add zero in sharedSizes for child nodes
	{
		sharedSizes[2 * tid] = 0;
		sharedSizes[2 * tid + 1] = 0;

	}

	__syncthreads();
	
	//copy all elements from shared matrices to the global memory ones for next level of tree
	if (!isLastLevel)
	{
		int index = tid;
		int stride = 1<<treeLevel; //stride used to avoid overwriting data when # of nodes are greater than blockDim.x

		for (int i = 0; i < 2; ++i)
		{
			sizeOfSequences[offset + i * stride] = sharedSizes[index];  //copy sizes

			startOfSequencesPoints[offset + i * stride] = sharedSequencesPoints[index]; //copy points' pointers
			startOfSequencesPointsAux[offset + i * stride] = sharedSequencesPointsAux[index];

			startOfSequencesIndexes[offset + i * stride] = sharedInd[index]; //copy indexes pointers
			startOfSequencesIndAux[offset + i * stride] = sharedIndAux[index];

			startOfSequencesDistances[offset + i * stride] = sharedSequencesDistances[index]; //copy distances store matrices
			startOfSequencesDistAux[offset + i * stride] = sharedSequencesDistancesAux[index];

			nodeLevelndex[offset + i * stride] = sharedLevelIndex[index]; //copy next level relative indexes

			index += blockDim.x; //update shared memory index

		}
	}

	cudaStreamDestroy(streams); //destroy cuda stream created by node

}
