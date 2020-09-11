
#include "vptree.h"

//---------------------------------------------------DEVICE KERNELS---------------------------------------------------------------------------------//

/*---matrix transpose kernel - used to transpose pointMatrix at the start of the tree creation for global memory access efficiency---//
-----------------------d1 and d2 represent the two dimensions of the matrix and matrix is the matrix to be transposed----------------//
-------------------processing of points assigned to each block is done in chunks to minimize shared memory usage and allow for more--//
-----------------------------------------blocks to run concurrently on the sms-------------------------------------------------------*/


__global__
void  matrixTranspose(double* oldMatrix, double* newMatrix, int d1, int d2, int blockPoints, int chunk)
{
	extern __shared__ double matrixPart[]; //shared memory table to hold points assigned to current block
										   //padding is not used since dimension of points acts as padding itself
										   //sizeof shared memory is chunk*d2

	int index = threadIdx.x + blockIdx.x * blockPoints * d2; //starting index for current thread and shared memory index
	int idxShared = threadIdx.x;

	int outputIndex = blockIdx.x * blockPoints + threadIdx.x; //index to dereference output table by each thread and bounf for subsequent for loop
	int boundOut = blockIdx.x * blockPoints + chunk;

	int bound = (blockIdx.x * blockPoints + chunk) * d2; //index bound for current block for each chunk
	int size = d1 * d2; //total size of array

	int counter = blockPoints / chunk; //blockPoints are assumed to be divisable by chunk

	for (int i = 0; i < counter; ++i) //done for the amount of chunks assigned to this block
	{
		for (; index < bound && index < size; index += matrixTransposeBlock) //copy all the points assigned to this block in shared memory
		{	                                                      //threads copy consecutive memory locations not just a single dimension
			matrixPart[idxShared] = oldMatrix[index];
			idxShared += matrixTransposeBlock;
		}

		__syncthreads(); //wait for all threads to complete writing to shared memory

		//output to newMatrix
		idxShared = threadIdx.x * d2; //calculate where each thread will start accessing shared memory

		for (; outputIndex < boundOut && outputIndex < d1; outputIndex += matrixTransposeBlock) //go through all dimensions of all points in current chunk
		{
			for (int y = 0; y < d2; ++y) //go through each dimension
			{
				newMatrix[outputIndex + y * d1] = matrixPart[idxShared++];
			}

			idxShared += (blockDim.x - 1) * d2; //move to next point in shared matrix assigned to this thread
		}

		//update bounds for next chunk - reset idxShared
		bound += chunk * d2;
		boundOut += chunk;
		idxShared = threadIdx.x;

		__syncthreads(); //complete part writing
	}
}


/*---matrix transpose kernel optimized for transposing memory into host friendly store manner meaning consecutive points instead of dimensions of points
--------------------------------in order to take advantage of cpu cache usage where coalescing isn't needed-------------------------------------------*/


__global__
void matrixTransposeToHost(double* oldMatrix, double* newMatrix, int d1, int d2, int blockPoints)
{
	extern __shared__ double matrixBuffer[]; //buffer to hold information before rewriting to global memory in order to have coalesced global memory access

	int index = threadIdx.x + blockIdx.x * blockPoints; //index for each thread to dereference oldMatrix
	int indexShared; //index for each thread to write to shared memory

	int outputIndex = blockIdx.x * blockPoints * d1 + threadIdx.x; //index for threadBlock to write in global memeory in -if possible- coalesced way

	for (int y = 0; y < d1; ++y) //for loop to read all dimensions of points assigned to this block
	{
		indexShared = threadIdx.x * d1 + y; //write current dimension of each point

		for (int i = index; i < d2 && i < (blockIdx.x + 1) * blockPoints; i += matrixTransposeBlock)
		{
			matrixBuffer[indexShared] = oldMatrix[i + y * d2];
			indexShared += matrixTransposeBlock * d1;
		}
	}

	//write transposed matrix to global memory
	__syncthreads();

	indexShared = threadIdx.x;

	for (; outputIndex < (blockIdx.x + 1) * blockPoints * d1 && outputIndex < d1 * d2; outputIndex += matrixTransposeBlock)
	{
		newMatrix[outputIndex] = matrixBuffer[indexShared];
		indexShared += matrixTransposeBlock;
	}

}


/*----kernel to copy data from global memory matrix to global memory again to assist in keeping an update matrix while partitioning---/
---------------------------size refers to the segment size while d1/d2 refer to the dimensions of the whole matrix-------------------*/

__global__
void copyKernel(double* pointsMatrixToCopy, double* pointsMatrixDest, int* indexMatrixToCopy, int* indexMatrixDest, int size, int d1, int d2, int partSize)
{
	int thid = threadIdx.x + blockIdx.x * partSize;  //dereferencing of matrices for each thread

	//copy index and double matrix
	for (int i = thid; i < size && i < partSize * (blockIdx.x + 1); i += BlockSize)
	{
		indexMatrixDest[i] = indexMatrixToCopy[i]; //copy index matrix
	}

	//done in two diferrent loops to maximize l1 use since matrices are processed in rows
	for (int y = 0; y < d2; ++y)
	{
		for (int i = thid; i < size && i < partSize * (blockIdx.x + 1); i += BlockSize)
			pointsMatrixDest[i + y * d1] = pointsMatrixToCopy[i + y * d1]; //copy points
	}

}

