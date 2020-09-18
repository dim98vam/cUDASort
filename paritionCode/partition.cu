#include "vptree.h"

/*-----------------device function to assist in parallel cumulative sum in partition kernel------------------*/
//only works with arrays whose number of elements is a power of two but since BlockSize is always a multiple of
//32 for efficiency this algorithm fits the purpose with no further modification - it is bank conflict free - uses
//more shared memory to achieve that but extra memory is negligible even in large blocks while performance gains are significant

__forceinline__ __device__
void parallelCumulative(int* matrixGreater, int* matrixLess, int* GlobalGreater, int* GlobalLess,
	int* sumLess, int* sumGreater, int thid, int* lessThanBoundary, int* greaterThanBoundary)
{

	unsigned int offset = 1; //initialize offset value used in tree traversal

	for (unsigned int d = BlockSize >> 1; d > 0; d >>= 1) // build sum in place up the tree - first stage of the cumulative sum
	{
		__syncthreads(); //to allow for all threads to have their data in place for next iteration

		if (thid < d) { //only half the threads or less are needed
			int ai = offset * (2 * thid + 1) - 1;  //calculate indices for shared matrix dereference
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);        //used for bank conflicts avoidance
			bi += CONFLICT_FREE_OFFSET(bi);

			matrixGreater[bi] += matrixGreater[ai]; //no thread sync is needed in this step since threads
			matrixLess[bi] += matrixLess[ai];       //access different elements in the matrix
		}

		offset *= 2; //update offset for next access
	}

	if (thid == 0) { // clear the last element in each matrix and update globals with the equivalent sum
		int padding = CONFLICT_FREE_OFFSET(BlockSize - 1);

		*sumGreater = atomicSub(GlobalGreater, matrixGreater[BlockSize - 1 + padding]) - matrixGreater[BlockSize - 1 + padding]; //greaterThan matrix operations                                          
		*greaterThanBoundary = matrixGreater[BlockSize - 1 + padding]; //update GreaterThanBoundary for later use in partitioning
		matrixGreater[BlockSize - 1 + padding] = 0;


		*sumLess = atomicAdd(GlobalLess, matrixLess[BlockSize - 1 + padding]); //lessThan matrix operations
		*lessThanBoundary = matrixLess[BlockSize - 1 + padding]; //update lessThanBoundary for later use in partitioning
		matrixLess[BlockSize - 1 + padding] = 0;
	}

	for (unsigned int d = 1; d < BlockSize; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1; //index calculation     
			int bi = offset * (2 * thid + 2) - 1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int t = matrixGreater[ai];   //greaterThan matrix operations 
			matrixGreater[ai] = matrixGreater[bi];
			matrixGreater[bi] += t;

			t = matrixLess[ai];                   //LessThan matrix operartions
			matrixLess[ai] = matrixLess[bi];
			matrixLess[bi] += t;

		}
	}

	__syncthreads(); //final thread sync
}


/*read/write Data functions - writes 64 bit data in shared memory in order to optimize global memory access in next step of partitioning -
-------------------------this is done to add data elements to shared memory and avoid global memory transactions------------------------*/

__forceinline__ __device__
void updateIndexes(double* globalMemory, double pivot, double* sharedBuffer, int* lessThan,
	int* greaterThan, int partSize, int size, int blockIndex, int thid, int counterMatrixIndex)
{
	double temp; //helper variable


	for (int i = thid + blockIndex * partSize; i < size && i < partSize * (blockIndex + 1);
		i += BlockSize)                 //iterate through the DataSet searching for elements less then or greater then pivot and filling shared buffers
	{                                   //size checked to make sure last block won't go out o bounds - grid is one dimensional
		temp = globalMemory[i];

		if (temp < pivot)
			++lessThan[counterMatrixIndex]; //increase count of less than pivot elements encountered by this thread
		else if (temp > pivot)
			++greaterThan[counterMatrixIndex]; //increase count of greater than pivot elements encountered by this thread

		sharedBuffer[i - blockIndex * partSize] = temp; //copy element into shared memory
	}
}


__forceinline__ __device__
void writeSharedMemory(double* dataMatrixShared, double pivot, double* sharedMatrix, int partSize,
	int size, int lessThan, int greaterThan, int blockIndex, int thid, int lessThanBoundary)
{
	int indexLess = lessThan;                        //shared memory conflicts are not addressed because indexing acts as random padding
	int indexGreater = greaterThan + lessThanBoundary; //and catering for conflicts would eventually be more expensive


	double temp; //helper variable

	//writing data to shared memory
	for (int i = thid + blockIndex * partSize; i < size && i < partSize * (blockIndex + 1);
		i += BlockSize)                                         //go through dataMatrix  adding elements to shared memory preload
	{
		temp = dataMatrixShared[i - blockIndex * partSize];

		if (temp < pivot)
		{
			sharedMatrix[indexLess] = temp; //place element less than pivot in shared memory
			++indexLess;
		}
		else if (temp > pivot)
		{
			sharedMatrix[indexGreater] = temp; //place element greater than pivot in shared memory
			++indexGreater;
		}
	}
}


template <typename T> //template to assist with fewer functions written

__forceinline__ __device__
void copySharedToGlobal(T* globalMatrix, T* sharedMatrix, int thid, int lessThanBoundary, int greaterThanBoundary,
	int sumLessIndex, int sumGreaterIndex)
{
	int sharedIndex = thid; //used for dereferencing sharedMatrix

	//copy first lessThan elements into global memory - copying is done in two phases beacause greaterThan and lessThan elements
	//may be far apart in global memory - and thus two seperate phases would result in better performance with as few transactions as possible

	for (int i = thid + sumLessIndex; sharedIndex < lessThanBoundary; i += BlockSize)
	{
		globalMatrix[i] = sharedMatrix[sharedIndex];
		sharedIndex += BlockSize;
	}

	sharedIndex = lessThanBoundary + thid; //update sharedIndex to access greaterThan elements in shared memory

	for (int i = thid + sumGreaterIndex; sharedIndex < greaterThanBoundary + lessThanBoundary; i += BlockSize)
	{
		globalMatrix[i] = sharedMatrix[sharedIndex];
		sharedIndex += BlockSize;
	}

}


template <typename U> //template to assist with fewer functions written

__forceinline__ __device__
void writeSharedMemoryData(double* dataMatrixShared, U* deviceData, double pivot, U* sharedMatrix, int partSize,
	int size, int lessThan, int greaterThan, int blockIndex, int thid, int lessThanBoundary)
{
	int indexLess = lessThan;                        //shared memory conflicts are not addressed because indexing acts as random padding
	int indexGreater = greaterThan + lessThanBoundary; //and catering for conflicts would eventually be more expensive


	double temp; //helper variable

	//writing data to shared memory
	for (int i = thid + blockIndex * partSize; i < size && i < partSize * (blockIndex + 1);
		i += BlockSize)                                         //go through dataMatrix  adding elements to shared memory preload
	{
		temp = dataMatrixShared[i - blockIndex * partSize];

		if (temp < pivot)
		{
			sharedMatrix[indexLess] = deviceData[i]; //place element less than pivot in shared memory
			++indexLess;
		}
		else if (temp > pivot)
		{
			sharedMatrix[indexGreater] = deviceData[i]; //place element greater than pivot in shared memory
			++indexGreater;
		}
	}
}




/*-----------------------------kernel to perform parallel partitioning on device - based on algorithm by tsigas/cederman----------------------------*/
/*----dataSet tables hold data to be partitioned - since partitioning is not done in place auxBuffer is used
  ----to hold data after partitioning - size indicates size of DataSet_Device table - Global_Less and Global_Greater hold
  ----values to indicate to each block where to write in auxBuffer_Device - update of the former is done atomically with fetch and add
  ----partSize indicates number of elements in dataSet to be cheched by each block - blockSync is used to track which block finished first
  ---- and assists in pivot placement - pivotIndex holds the final index of the pivot */


__global__ void
partitionKernel(double* DataSet_Device, double* auxBuffer_Device, double* pIvot, int* pivotIndex, int size, int partSize,
	int* Global_Less, int* Global_Greater, int* blockSync, int* index_Device, int* indexAux, double* pointsDevice,
	double* pointsAux, int d1, int d2, int pivotPlace)
{
	__shared__ int lessThan[sharedMatrixSize];    //matrices to hold count of elements found greater or less than the pivot
	__shared__ int greaterThan[sharedMatrixSize]; //encoutered by each thread in the block - shared memory used for speed of access
	__shared__ int sumLess;    //variables to hold running sum value
	__shared__ int sumGreater; //beggining from zero because sum is exclusive - stored in shared memory to later be used by all threads concurrently

	__shared__ double sharedBuffer[partitionPartSize]; //shared memory matrix to hold non partitioned block points for every subsequent pass

	__shared__ double sharedPreload[partitionPartSize];   //shared memory matrices to hold partitioned block points before writing them to auxBuffer

	__shared__ int lessThanBoundary; //holds numberOfElements found less than pivot for this block - used in shared memory dereferencing
	__shared__ int greaterThanBoundary; //holds numberOfElements found less than pivot for this block - used in shared memory dereferencing

	double pivot = *(pIvot); //register to hold pivot value for each thread

	int tid = threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x); //thread index with padding to avoid shared memory bank conflicts


	if (threadIdx.x == 0) //initialize sumLess and SumGreater variables
	{
		sumLess = 0;
		sumGreater = 0;
	}

	lessThan[tid] = 0; //initialize matrices
	greaterThan[tid] = 0;

	updateIndexes(DataSet_Device, pivot, sharedBuffer, lessThan, greaterThan, partSize, size, blockIdx.x, threadIdx.x, tid); //perform first pass and copy data into shared memory
																													//for future use - also update less/greaterThan matrices


	//first pass of DataSet completed - accumulative sum of elements in shared matrices done now to calculate
	//where each thread and block should write in the auxBuff - some block synchronization needed now for global cumulative sum in 
	//global_less and global_greter variables

	__syncthreads(); //synchronize block threads to perform cumulative sum

	parallelCumulative(greaterThan, lessThan, Global_Greater, Global_Less, &sumLess, &sumGreater, threadIdx.x, &lessThanBoundary,
		&greaterThanBoundary); //perform cumulative sum in parallel


//using previously initialized shared memory place elements in sharedPreload in partitioned manner
	writeSharedMemory(sharedBuffer, pivot, sharedPreload, partSize, size, lessThan[tid], greaterThan[tid], blockIdx.x, threadIdx.x, lessThanBoundary);

	__syncthreads(); //synchronize threads to ensure sharedMatrixPreload has been written

	//scan sharedPreload to copy elements in global memory auxBuffer in a coalesced way
	copySharedToGlobal<double>(auxBuffer_Device, sharedPreload, threadIdx.x, lessThanBoundary, greaterThanBoundary, sumLess, sumGreater);

	//after partitioning distance data rearrange index and point data accordingly
	__syncthreads();

	//indexes' part
	writeSharedMemoryData<int>(sharedBuffer, index_Device, pivot, (int*)sharedPreload, partSize, size, lessThan[tid], greaterThan[tid],
		blockIdx.x, threadIdx.x, lessThanBoundary);
	__syncthreads();

	copySharedToGlobal<int>(indexAux, (int*)sharedPreload, threadIdx.x, lessThanBoundary, greaterThanBoundary, sumLess, sumGreater);



	//points' part
	for (int y = 0; y < d2; ++y)
	{
		__syncthreads();
		writeSharedMemoryData<double>(sharedBuffer, pointsDevice + y * d1, pivot, sharedPreload, partSize, size, lessThan[tid], greaterThan[tid],
			blockIdx.x, threadIdx.x, lessThanBoundary);

		__syncthreads();

		copySharedToGlobal<double>(pointsAux + y * d1, sharedPreload, threadIdx.x, lessThanBoundary, greaterThanBoundary, sumLess, sumGreater);
	}


	//element placement is complete - update blockSync variable to indicate this block is done - last block also places the pivot in right place
	//and returns the index to the kernel caller

	if (threadIdx.x == 0) //no __syncthreads command is used since block synchronization doesn't require all threads of each block to be done
						  //only threadId 0 beacause then the atomic add on sumLess done by this block is for sure complete
	{
		unsigned int isDone = atomicSub(blockSync, 1);
		if (isDone == 1) //this is the last block to complete
		{
			auxBuffer_Device[*Global_Less] = pivot; //place pivot in Global_Less position since this is past the last position 
													//where less than elements where added (all blocks added to this variable)
			*pivotIndex = (*Global_Less) + 1; //return pivotIndex

			//place index of pivot and point corrsponding to that pivot in correct position as well
			indexAux[*Global_Less] = index_Device[pivotPlace - 1]; //index part

			for (int y = 0; y < d2; ++y)  //account for every dimension in pointPivot - points part
				pointsAux[y * d1 + (*Global_Less)] = pointsDevice[y * d1 + pivotPlace - 1];
		}
	}
}


/*-----------kernel for distance calculation - matrix with points is stored in column  major way thus if n points of d dimensions exist
-------------the stored matrix is dxn - this is done to allow for parallelism in distance calculation while achieving coalesced memory
---------------------------------------access - totalPoints refers to the d1 of the matrix and is used for dereferencing-------------*/

__global__ void
distanceCalculation(double* pointMatrix, double* distanceMatrix, int numOfpoints, int dimensions, double* pivotPoint,
	int partSize, int totalPoints) //partSize holds size of subTable processed by each block 
{
	extern __shared__ double pivot[]; //table in shared memory to hold pivot point since many accesses are made to it
	int tid = threadIdx.x + blockIdx.x * partSize; //index to indicate where thread should write in distance matrix - 
														  //also used in for loop to help traverse the pointMatrix


	if (threadIdx.x == 0) //let tid 0 copy pivot in shared memory - slow operation but beneficial afterwards
	{
		for (int i = 0; i < dimensions; ++i)
			pivot[i] = *(pivotPoint + i * totalPoints);
	}

	//boundaries used in subsequent for loops
	int secondBound = (blockIdx.x + 1) * partSize;

	for (int y = tid; y < secondBound && y < numOfpoints; y += distanceBlockSize) //initialize distance matrix - done outside of main for loop
		distanceMatrix[y] = 0.0;                                         //to avoid unnecessary if statements


	__syncthreads();

	for (unsigned int i = 0; i < dimensions; ++i) //for loop to traverse through all rows of pointMatrix
	{
		for (int y = tid; y < secondBound && y < numOfpoints; y += distanceBlockSize) //traverse through current row of table - index is such that bound do need
																			  //further multiplication
		{
			double temp = (pointMatrix[y + i * totalPoints] - pivot[i]) * (pointMatrix[y + i * totalPoints] - pivot[i]); //calculate squares and add to distance matrix position
																														//multiplication instead of pow used for speed
			distanceMatrix[y] += temp;
		}
	}

	__syncthreads();

	for (int y = tid; y < secondBound && y < numOfpoints; y += distanceBlockSize) //calculate square root
		distanceMatrix[y] = sqrt(distanceMatrix[y]);


}


