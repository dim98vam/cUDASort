#include "knnSearch.h"
#include <chrono>
#include <iostream>


__inline__ __device__
double calcDist(double* vp, double* queryPoint, int d2)
{
    double dist2 = 0;

    for (int i = 0; i < d2; i++) {
        dist2 += (vp[i] - queryPoint[i]) * (vp[i] - queryPoint[i]);
    }
    return sqrt(dist2);
}

__inline__ __device__
void insert(double distance, int index, int* indexMatrix, double* distanceMatrix, int k)
{
    int i;
    for (i = k - 2; (i >= 0 && distanceMatrix[i] > distance); i--)
    {
        distanceMatrix[i + 1] = distanceMatrix[i]; //swap elements so that greater is at the end of the list
        indexMatrix[i + 1] = indexMatrix[i];
    }

    distanceMatrix[i + 1] = distance; //place element in correct position
    indexMatrix[i + 1] = index;

}

__global__
void searchKernel(double* points, double* md, int* indexMatrix, uint8_t* leftChild, uint8_t* rightChild, int k, int numberOfPoints, int d2, double* query,
    double* globalDistances, int* globalIndexes, int numberOfQuery)
{
    //shared memory allocation

    int holder = (int)floor(log2((double)numberOfPoints)) + 1; //helper variable

    extern __shared__ int nodeStack[];

    int* stack = nodeStack + holder * threadIdx.x; //array to act as stack holding previously visited nodes
    int* visited = ((int*)(stack + holder * (blockDim.x - threadIdx.x))) + threadIdx.x * holder; //array to hold whether left or right child was visited last
    double* distanceStack = ((double*)(visited + holder * (blockDim.x - threadIdx.x))) + threadIdx.x * holder; //holds distances of already visited nodes

    int* index = ((int*)(distanceStack + holder * (blockDim.x - threadIdx.x))) + threadIdx.x * k;  //array to hold indexes of nearest neighbors
    double* distances = ((double*)(index + (blockDim.x - threadIdx.x) * k)) + threadIdx.x * k; //array to hold distances of nearest neighbors

    int* exists = ((int*)(distances + (blockDim.x - threadIdx.x) * k)) + threadIdx.x * holder; //array to hold whether other child of node exists 

  

    int tid = blockIdx.x * blockDim.x + threadIdx.x; //query point of current thread
    int dummy;
    int stackEnd;
    double distHolder;
    double* queryHolder;
    int visitedHolder;

    if (tid < numberOfQuery) //to account for number of query points non divisable by blockDim.x
    {

        //initializing stack pointers and distances matrix
        stackEnd = 0;
        stack[stackEnd] = 0; //stackEnd points one element after the last element indicating how many elements exist in the stack
        visited[stackEnd] = 0;
        
        dummy = 0;
        
        while(dummy < k)
        {
            distances[dummy] = DBL_MAX; //initialize distances to infinity
            dummy+=1;
        }

        queryHolder = query + blockIdx.x * blockDim.x * d2 + threadIdx.x * d2;


        while (stackEnd > -1)
        {

            visitedHolder = visited[stackEnd];

            if (visitedHolder != 0) //node already visited once
            {

                holder = stack[stackEnd]; //pop current node from stack since it is not needed
               

                if (visitedHolder == 1 && exists[stackEnd] && md[holder] < (distanceStack[stackEnd] + distances[k - 1]))  //if left node was visited and right node exists
                {

                    stack[stackEnd] = holder * 2 + 2; //add right child node in stack for next iteration
                    visited[stackEnd] = 0;

                    continue; //continue to next iteration

                }
                else if (visitedHolder == 2 && exists[stackEnd] && (md[holder] + distances[k - 1]) > distanceStack[stackEnd])
                {
                    stack[stackEnd] = holder * 2 + 1; //add left child to stack
                    visited[stackEnd] = 0;

                    continue; //continue to nect iteration
                }

                --stackEnd; //if no element was added pop this from stack

            }
            else
            {
               
                distHolder = calcDist(points + stack[stackEnd] * d2, queryHolder, d2); //calculate distance between vp and query points for current thread
                distanceStack[stackEnd] = distHolder;

                holder = stack[stackEnd];

                if (distHolder < distances[k - 1]) //if current point is closer to query than previous closest
                    insert(distHolder, indexMatrix[holder], index, distances, k); //update closest neighbors' distances and indexes - done sequentially for small number of neighbors

                if (distHolder <= md[holder]) //search left subtree if it exists
                {
                    visited[stackEnd] = 1;

                    if (rightChild[holder])
                        exists[stackEnd] = 1;
                    else
                        exists[stackEnd] = 0;

                    if (leftChild[holder])
                    {
                        ++stackEnd; //create space for child without popping father
                        
                        stack[stackEnd] = holder * 2 + 1;
                        visited[stackEnd] = 0;

                        continue;
                    }

                }
                else if (distHolder > md[holder]) //search right subtree
                {

                    visited[stackEnd] = 2;

                    if (leftChild[holder])
                        exists[stackEnd] = 1;
                    else
                        exists[stackEnd] = 0;


                    if (rightChild[holder])
                    {
                        ++stackEnd; //create space for child without popping father
                        
                        stack[stackEnd] = holder * 2 + 2;
                        visited[stackEnd] = 0;

                        continue;
                    }
                }

            }
        }

        //write shared memory to global in column major format
        holder = 0; //index to dereference shared memory

        while (holder < k)
        {
            globalIndexes[tid + numberOfQuery * holder] = index[holder];      //copy elements
            globalDistances[tid + numberOfQuery * holder] = distances[holder];

            ++holder; //read next neighbor
        }
    }
}


//wrapper function to call kernel above from host code

knnresult* searchQuery(vptree* root, int k, int numberOfPoints, int d2, double* query, int numberOfQuery)
{
    int numOfBlocks = (int)(numberOfQuery + BlockSizeQuery - 1) / BlockSizeQuery; //number of blocks to call to cover all query points

    //cpu pointers
    double* Distances;
    int* Indexes;

    Distances = (double*)malloc(numberOfQuery * k * sizeof(double));
    Indexes = (int*)malloc(numberOfQuery * k * sizeof(int));

    //allocate memory on gpu

    //return values
    double* globalDistances;
    int* globalIndexes;

    cucheck_dev(cudaMalloc((void**)&globalDistances, numberOfQuery * k * sizeof(double)));
    cucheck_dev(cudaMalloc((void**)&globalIndexes, numberOfQuery * k * sizeof(int)));

    //tree table values
    double* points;
    double* md;
    int* indexMatrix;
    uint8_t* leftChild;
    uint8_t* rightChild;
    double* queryDevice;


    cucheck_dev(cudaMalloc((void**)&points, numberOfPoints * d2 * sizeof(double)));
    cucheck_dev(cudaMalloc((void**)&md, numberOfPoints * sizeof(double)));
    cucheck_dev(cudaMalloc((void**)&indexMatrix, numberOfPoints * sizeof(int)));
    cucheck_dev(cudaMalloc((void**)&queryDevice, numberOfQuery * d2 * sizeof(double)));
    cucheck_dev(cudaMalloc((void**)&leftChild, numberOfPoints * sizeof(uint8_t)));
    cucheck_dev(cudaMalloc((void**)&rightChild, numberOfPoints * sizeof(uint8_t)));


    cudaMemcpy(points, root->points, numberOfPoints * d2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(md, root->md, numberOfPoints * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(indexMatrix, root->indexes, numberOfPoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(leftChild, root->leftChild, numberOfPoints * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(rightChild, root->rightChild, numberOfPoints * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(queryDevice, query, numberOfQuery * d2 * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    int holder = (int)floor(log2(numberOfPoints)) + 1;

    size_t sharedSize = holder * BlockSizeQuery * (sizeof(int) + 2 * sizeof(int) + sizeof(double)) + BlockSizeQuery * k * (sizeof(int) + sizeof(double));
    auto start = std::chrono::high_resolution_clock::now(); //start timing algorithm

    searchKernel << <numOfBlocks, BlockSizeQuery, sharedSize >> > (points, md, indexMatrix, leftChild, rightChild, k, numberOfPoints, d2, queryDevice, globalDistances,
                                                                   globalIndexes, numberOfQuery);

    cucheck_dev(cudaGetLastError());

    cucheck_dev(cudaDeviceSynchronize()); //wait for search kernel to be done

    auto stop = std::chrono::high_resolution_clock::now(); //record end of algorithm
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    printf("Neighbour search took: ");
    std::cout << duration.count();

    printf(" us\n\n");


    cucheck_dev(cudaMemcpy(Distances, globalDistances, numberOfQuery * k * sizeof(double), cudaMemcpyDeviceToHost));
    cucheck_dev(cudaMemcpy(Indexes, globalIndexes, numberOfQuery * k * sizeof(int), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    knnresult* result = (knnresult*)malloc(sizeof(knnresult));

    result->nidx = Indexes;
    result->ndist = Distances;
    result->k = k;
    result->m = numberOfQuery;


    //free gpu memory
    cudaFree(globalDistances);
    cudaFree(globalIndexes);
    cudaFree(points);
    cudaFree(md);
    cudaFree(leftChild);
    cudaFree(rightChild);
    cudaFree(queryDevice);

    return result;

}
