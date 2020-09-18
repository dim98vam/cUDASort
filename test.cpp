// =========================================================
// === STACK IMPLEMENTATION TO TEST RECUSRION INVARIANCE ===
// =========================================================

// Declare linked list node 


#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <assert.h>
#include "vptree.h"
#include "knnSearch.h"
#include "tester_helper.h"


typedef struct node {
	vptree* T;
	int isInner;
	double md;
	struct node* link;
} node;

//#define VERBOSE

/*static char* STR_CORRECT_WRONG[] = { "WRONG", "CORRECT" };



// Utility function to add an element data in the stack 
// insert at the beginning 
void push(node** top, vptree* T, double md, int isInner)
{
	// create new node temp and allocate memory 
	node* temp = (node*)malloc(sizeof(node));

#ifdef VERBOSE
	fprintf(stdout, "PUSH ENTERED\n");
#endif

	// check if stack (heap) is full. Then inserting an element would 
	// lead to stack overflow 
	if (!temp) {
		fprintf(stderr, "\npush: Heap Overflow\n");
		exit(1);
	}

	// initialize data into temp data field 
	temp->T = T;
	temp->isInner = isInner;
	temp->md = md;

	// put top pointer reference into temp link 
	temp->link = *top;

	// make temp as top of Stack 
	*top = temp;

}

// Remove top element from stack
void pop(node** top)
{

#ifdef VERBOSE
	fprintf(stdout, "POP ENTERED\n");
#endif

	// check for stack underflow 
	if (*top == NULL) {
		fprintf(stderr, "\npop: Stack Underflow\n");
		exit(1);
	}

	// top assign into temp 
	struct node* temp = *top;

	// return second as top
	*top = (*top)->link;

	// destroy connection between first and second 
	temp->link = NULL;

	// release memory of top node 
	free(temp);

}




// =================
// === UTILITIES ===
// =================

double dist(double* X, double* Y, int d) {
	double dist2 = 0;
	for (int i = 0; i < d; i++) {
		dist2 += (X[i] - Y[i]) * (X[i] - Y[i]);
	}
	return sqrt(dist2);
}

// Function to print all the  
// elements of the stack  
int verifyLeafPlace(node** top, double* X, int d)
{


	// check for stack underflow 
	if (*top == NULL) {
		fprintf(stderr, "\nverifyLeafPlace: Stack Underflow\n");
		exit(1);
	}

	struct node* temp = *top;

	// iterate the ancestors in stack
	while (temp != NULL) {

#ifdef VERBOSE
		fprintf(stdout, "%f | %f | %d == %d\n", dist(X, getVP(temp->T), d), temp->md, temp->isInner, dist(X, getVP(temp->T), d) <= temp->md);
#endif

		// check whether point should be inside or outside
		int isInner = dist(X, getVP(temp->T), d) <= temp->md;

		// if the direction is invalid, break and return false
		if (isInner != temp->isInner) 
			return 0;

		// assign temp link to temp 
		temp = temp->link;
	}

	return 1;

}




// ==================
// === VALIDATION ===
// ==================

int* foundInTree;

int verifyTree(vptree* T, double* vp, node** stack, double md, int isInner,
	int n, int d) {

	int isValid = 1;

	// if empty, return
	if (T == NULL) return isValid;

	int isValidAncestor = 1;
	int isLeaf = (getInner(T) == NULL && getOuter(T) == NULL);

#ifdef VERBOSE
	fprintf(stdout, "%x %x\n", stack, *stack);
#endif  

	// if leaf check ancestor path
	if (isLeaf) isValidAncestor = verifyLeafPlace(stack, getVP(T), d);
	


	// if inner, radius must be smaller than parent's diameter
	if (isInner && getMD(T) > 2 * md) return 0;

	// update list of indices
	int idx = getIDX(T);
	if (idx < n) {
		foundInTree[idx] = 1;
	}
	else
		printf("found index outOfBounds\n");

	// validate distance to parent
	if (isInner) {
		isValid = dist(vp, getVP(T), d) <= md;
		if (!isValid)
			printf("inner issue\n");
	}
	else {
		isValid = dist(vp, getVP(T), d) > md;
	}

	// recurse if not leaf
	if (!isLeaf) {

		// add to stack as inner and recurse, then pop
		push(stack, T, getMD(T), 1);
		int isValidInn = verifyTree(getInner(T), getVP(T), stack, getMD(T), 1, n, d);
		pop(stack);

		// add to stack as outer and recurse, then pop
		push(stack, T, getMD(T), 0);
		int isValidOut = verifyTree(getOuter(T), getVP(T), stack, getMD(T), 0, n, d);
		pop(stack);

		// all conditions must be true
		isValid = isValid && isValidInn && isValidOut;

	}
	else {

		// make sure ancestory path is correct
		isValid = isValid && isValidAncestor;

	}

	// return 
	return isValid;

}*/



int main()
{

	int n =12400;//data
	int d = 3;//dimensions

	double* dataArr = (double*)malloc((n * d) * sizeof(double));
	//double* zeros = (double*)calloc(d, sizeof(double));

	//foundInTree = (int*)calloc(n, sizeof(int));
	srand(5);
	
	for (int i = 0; i < n * d; i++)
		dataArr[i] = rand() % 300000;
	

	//double dataArr[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 };

	vptree* root = buildvp(dataArr, n, d);

	/*node* stack = NULL;

	int isValid = verifyTree(root, zeros, &stack, INFINITY, 1, n, d);
	int foundAll = 1;

	assert(stack == NULL);

	for (int i = 0; i < n; i++)
		if (!foundInTree[i]) {
			foundAll = 0;
			break;
		}

	printf("Tester validation: %s PROPERTIES | %s INDICES\n",
		STR_CORRECT_WRONG[isValid], STR_CORRECT_WRONG[foundAll]);*/

	int* indexes = (int*)malloc(n*3*sizeof(int));
	double* distances = (double*)malloc(n*3*sizeof(double));


	int temp = ((int)floor(log2(n))) + 1;
	int sizeHolder = (1 << temp)-1;
	
	knnresult knnres = *searchQuery(root, 3, sizeHolder, d , dataArr, n);

	for (int i = 0; i < n; ++i)
	{
		for (int y = 0; y < 3; ++y)
		{
			indexes[i * 3 + y] = knnres.nidx[y * n + i];
			distances[i * 3 + y] = knnres.ndist[y * n + i];
		}
	}
	
	knnres.nidx = indexes;
	knnres.ndist = distances;
	
	
	
	
	int isValidC = validateResult(knnres, dataArr, dataArr, n, n, d, 3, COLMAJOR);

	int isValidR = validateResult(knnres, dataArr, dataArr, n, n, d, 3, ROWMAJOR);

	printf("Tester validation: %s NEIGHBORS\n",
		STR_CORRECT_WRONG[isValidC || isValidR]);

	

	return 0;

}
