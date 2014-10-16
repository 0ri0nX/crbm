#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>
#include <math.h>
#include <signal.h>
using namespace std;
#include "matrix.h"
using namespace YAMATH;

#define BLOCK_SIZE 32
#define CUBLAS
#define ERR
/*
#define BP
#define FP
*/

/* Forward declarations */
class NeuralTopology;
class MatrixCpuAsync;
template<typename Matrix> class MatrixAsync;
template<typename Matrix> class OperationAsyncGpu;
template<typename Matrix> class OperationMatrixAsync;
template<typename Matrix> class OperationMatrixMultiplyAsync;

enum Operation{
    SubErrorCountOperation,
    AddOperation,
    ReduceOperation,
    ActivateOperation,
    ConstMultOperation,
    DeltaNonlinOperation,
    EpsOperation,
    WeightOperation,
};

typedef struct {
    int width;
    int height;
    float* elements;
    int stride;
} MatrixPointer;


/*  Kernels Declarations */
__global__ void MatMulKernel(MatrixPointer A, MatrixPointer B, MatrixPointer C , 
        int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);
__global__ void ActivKernel(MatrixPointer A, MatrixPointer B, int size);
__global__ void MatSubKernel(MatrixPointer A, MatrixPointer B, MatrixPointer C, int size);
__global__ void MatAddKernel(MatrixPointer A, MatrixPointer B, MatrixPointer C, int size);
__global__ void MatReduceKernel(MatrixPointer A, MatrixPointer B);
__global__ void MatMulConstKernel(MatrixPointer A, MatrixPointer B, float constant, int size);
__global__ void MatNonlinDleta(MatrixPointer Delta, MatrixPointer Y, MatrixPointer Result,  int size);
__global__ void EpsKernel(MatrixPointer lastDir, MatrixPointer actDir, MatrixPointer lSpeed, int size);
__global__ void WeightKernel(MatrixPointer W, MatrixPointer Eps, MatrixPointer Result,  int size);


/******************************************************************************
 ******************************************************************************/


template<typename Matrix> class OperationAsyncGpu{
    public:
        OperationAsyncGpu(void){}
        virtual void Execute(MatrixAsync<Matrix> &outMatrix, cublasHandle_t inHandle, cudaStream_t stream) const = 0;
        virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const = 0;
        virtual ~OperationAsyncGpu(void){}

    protected:
        static const float m_Zero;
        static const float m_One;
        static const float m_MinusOne;
};

template<typename Matrix>  const float OperationAsyncGpu<Matrix>::m_Zero = 0.0f;
template<typename Matrix>  const float OperationAsyncGpu<Matrix>::m_One = 1.0f;
template<typename Matrix>  const float OperationAsyncGpu<Matrix>::m_MinusOne = -1.0f;

/* Async operations */
template<typename Matrix> 
class OperationMatrixMultiplyAsync: public OperationAsyncGpu<Matrix>{
    public:
        OperationMatrixMultiplyAsync(
                const MatrixAsync<Matrix>& inA, 
                const MatrixAsync<Matrix>& inB);
        virtual~OperationMatrixMultiplyAsync();
        virtual void Execute(
                MatrixAsync<Matrix> &outMatrix, 
                cublasHandle_t inHandle,
                cudaStream_t stream) const;
        virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const;

    protected:
        const MatrixAsync<Matrix>& m_A;
        const MatrixAsync<Matrix>& m_B;
};


template<typename Matrix> 
class OperationMatrixAsync: public OperationAsyncGpu<Matrix>{
    public:
        OperationMatrixAsync(
                const MatrixAsync<Matrix>& inA, 
                const MatrixAsync<Matrix>& inB, 
                Operation operationType);
        
        OperationMatrixAsync(
                const MatrixAsync<Matrix>& inA, 
                const float constant, 
                Operation operationType);

        virtual~OperationMatrixAsync(){}
        virtual void Execute(
                MatrixAsync<Matrix> &outMatrix, 
                cublasHandle_t inHandle,
                cudaStream_t stream) const;
        virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const { outX = 1;};

    protected:
        const MatrixAsync<Matrix>& m_A;
        const MatrixAsync<Matrix>& m_B;
        const float constant;
        Operation operationType;
};

/* Async wrapper for GPU Matrix */
template<typename Matrix> 
class MatrixAsync{
    public:
        MatrixAsync(void);
        MatrixAsync(int x, int y, bool inTransposed);
        ~MatrixAsync();
        void RandNormal(float inMean, float inStdDev, unsigned long long inSeed = 0);
        OperationMatrixMultiplyAsync<Matrix> operator*(const MatrixAsync<Matrix>& inB) const;
        OperationMatrixAsync<Matrix>  operator*(const float constant) const;
        OperationMatrixAsync<Matrix> operator-(const MatrixAsync<Matrix>& inB) const;
        OperationMatrixAsync<Matrix> operator+(const MatrixAsync<Matrix>& inB) const;
        OperationMatrixAsync<Matrix> Reduce(void) const;
        OperationMatrixAsync<Matrix> Activate(void) const;
        OperationMatrixAsync<Matrix> DeltaNonlinCount(const MatrixAsync<Matrix>& inB) const;
        void EpsRefreshInSitu(const MatrixAsync<Matrix>& deltaOld, const MatrixAsync<Matrix>& deltaNew);
        void WeightRefreshInSitu(const MatrixAsync<Matrix>& eps);
        void DeltaNonlinCountInSitu(const MatrixAsync<Matrix>& inB);
        void ReduceInSitu(void);
        void ActivateInSitu(void);
        void operator=(const OperationAsyncGpu<Matrix>& inOperation);
        MatrixAsync<Matrix>& operator=(const MatrixAsync<Matrix> &inMatrix);
        MatrixAsync<Matrix>& operator=(const MatrixCpu &inMatrix);
        MatrixAsync<Matrix>& operator[](int idx);
        

        const Matrix& getMatrixConst(void) const;
        Matrix& getMatrix(void);

        float* getDataConst(void) const;
        float* getDataPatchConst(void) const;
        float* getData(void);
        float* getDataPatch(void);

        int getX(void) const;
        int getY(void) const;
        int getPatchX(void) const;
        int getPatchY(void) const;
        int getPatchPosX(void) const;
        int getPatchPosY(void) const;

        void Reset(int x, int y);
        bool isTrans(void) const;
        void Transpose(void);
        bool isPatched(void) const;
        bool setPatch(int row, int col, int rowSize, int colSize);
        bool movePatch();

    protected:
        cudaStream_t stream;
        cublasHandle_t handle;
        Matrix matrix;
        bool patch;
        int startPosition;
        int rowPos;
        int colPos;
        int rowSize;
        int colSize;
};


template<int N>
class Error{
    public:
        Error(int tolerance);
        ~Error();
        void next(float value);
        void loadErr(float value);
        void loadW(MatrixAsync<MatrixGpu> (&w)[N]);
        bool finished(void);
        bool loadNewW(void);
        bool loadOldW(void);
        float oldValue;

        MatrixAsync<MatrixGpu> w[N];

    protected:
        bool runOnTolerance;
        int tolerance;
        int actualTolerance;
        bool oW;
        bool nW;
        bool fin;

};

template<int N>
Error<N>::Error(int tolerance): 
        tolerance(tolerance), 
        actualTolerance(tolerance),
        fin(false),
        oW(false),
        nW(false),
        runOnTolerance(false),
        oldValue(FLT_MAX)
        {}

template<int N>
Error<N>::~Error(){}

template<int N>
bool Error<N>::loadNewW(){
    return nW;
}

template<int N>
bool Error<N>::finished(){
    return fin;
}

template<int N>
bool Error<N>::loadOldW(){
    return oW;
}

template<int N>
void Error<N>::loadW(MatrixAsync<MatrixGpu> (&w)[N]){
    for(int i=0; i<N; i++)
        this->w[i] = w[i];
    oW = false;
    nW = false; 
}

template<int N>
void Error<N>::next(float value){
    if(value>oldValue){
        if(runOnTolerance){
            if(--actualTolerance < 0)
                fin = true;
        }
        else{
           runOnTolerance = true; 
           oW = true;
        } 
        
    }
    else{
        if(runOnTolerance){
            actualTolerance = tolerance;
            nW = true;
        }
        oldValue = value;
    }

}


/* MatrixAsync methods
 ******************************************************************************
 ******************************************************************************/


template<typename Matrix> 
void MatrixAsync<Matrix>::RandNormal(float inMean, float inStdDev, unsigned long long inSeed){
    matrix.RandNormal(inMean, inStdDev, inSeed);
}


template<typename Matrix> 
void MatrixAsync<Matrix>::Transpose(void){
    matrix.Transpose();
}

template<typename Matrix> 
void MatrixAsync<Matrix>::Reset(int x, int y){
    matrix.Reset(x,y);
    patch = false;
    startPosition = 0;
    rowSize = 0;
    rowPos = 0;
    colSize = 0;
    colPos = 0;
}

template<typename Matrix> 
bool MatrixAsync<Matrix>::movePatch(){
    if(!patch) return false;
    return setPatch(this->rowPos+this->rowSize, colPos, this->rowSize, this->colSize);
}

template<typename Matrix> 
bool MatrixAsync<Matrix>::isPatched() const{
    return patch;
}

template<typename Matrix> 
bool MatrixAsync<Matrix>::setPatch(int row, int col, int rowSize, int colSize){
    int mr = matrix.getX();
    int mc = matrix.getY();

    if(row>=mr) return false;
    if(col>=mc) return false;
    patch = true;
    rowPos = row;
    colPos = col;
    startPosition = col*mr + row;
    if((row+rowSize)<mr)
        this->rowSize = rowSize;
    else this->rowSize = mr-row;
    if((col+colSize)<mc)
        this->colSize = colSize;
    else this->colSize = mc - col; 
    return true;
}

/* MatrixAsync methods */
template<typename Matrix> 
MatrixAsync<Matrix>::MatrixAsync(void):
        patch(false), 
        startPosition(0){
    cudaStreamCreate( &stream);
    cublasCreate(&handle);
}

template<typename Matrix> 
MatrixAsync<Matrix>::MatrixAsync(int x, int y, bool inTransposed = false):
        matrix(x,y,inTransposed), 
        patch(false), 
        startPosition(0){
    cudaStreamCreate( &stream);
    cublasCreate(&handle);
}

template<typename Matrix> 
MatrixAsync<Matrix>::~MatrixAsync(){
    cudaStreamDestroy(stream);
    cublasDestroy(handle);
}

template<typename Matrix> 
const Matrix& MatrixAsync<Matrix>::getMatrixConst(void) const{
    return this->matrix;
} 

template<typename Matrix> 
Matrix&  MatrixAsync<Matrix>::getMatrix(void){
    return this->matrix;
}

template<typename Matrix> 
float* MatrixAsync<Matrix>::getDataConst(void) const{
    return matrix.getDataConst();
 
}

template<typename Matrix> 
float* MatrixAsync<Matrix>::getDataPatchConst(void) const{
   return &matrix.getDataConst()[startPosition];
}

template<typename Matrix> 
float* MatrixAsync<Matrix>::getData(void){
    return matrix.getData();
}

template<typename Matrix> 
float* MatrixAsync<Matrix>::getDataPatch(void){
    return &matrix.getData()[startPosition];
}

template<typename Matrix> 
int MatrixAsync<Matrix>::getX(void) const{
    return matrix.getX();
}

template<typename Matrix> 
int MatrixAsync<Matrix>::getY(void)const {
    return matrix.getY();
}

template<typename Matrix> 
int MatrixAsync<Matrix>::getPatchX(void) const{
    if(patch) return rowSize;
    else return matrix.getX();
}

template<typename Matrix> 
int MatrixAsync<Matrix>::getPatchY(void)const {
    if(patch) return colSize;
    else return matrix.getY();
}

template<typename Matrix> 
int MatrixAsync<Matrix>::getPatchPosX(void) const{
    if(patch) return rowPos;
    else return 0;
}

template<typename Matrix> 
int MatrixAsync<Matrix>::getPatchPosY(void)const {
    if(patch) return colPos;
    else return 0;
}

template<typename Matrix> 
bool MatrixAsync<Matrix>::isTrans(void)const {
    return matrix.isTrans();
}

/* MatrixMultiplyOperation methods                                            *
 ******************************************************************************/
template<typename Matrix> 
OperationMatrixMultiplyAsync<Matrix>::OperationMatrixMultiplyAsync(
        const MatrixAsync<Matrix>& inA, 
        const MatrixAsync<Matrix>& inB): 
        m_A(inA), 
        m_B(inB){}

template<typename Matrix> 
OperationMatrixMultiplyAsync<Matrix>::~OperationMatrixMultiplyAsync(){}

template<typename Matrix> 
void OperationMatrixMultiplyAsync<Matrix>::GetResultSize(
        int &outX, 
        int &outY, 
        bool &outTransposed) const{
    outX = m_A.getPatchX();
    outY = m_A.getPatchY();
    outTransposed = m_A.isTrans();
}

template<typename Matrix> 
void OperationMatrixMultiplyAsync<Matrix>::Execute(
    MatrixAsync<Matrix> &outMatrix, 
    cublasHandle_t inHandle,
    cudaStream_t stream) const{

	    MatrixPointer d_A, d_B, d_C; 
	    d_A.width  = m_A.getPatchY();
	    d_A.height = m_A.getPatchX(); 
	    d_A.stride = m_A.getX();
	    d_A.elements = m_A.getDataPatchConst();
	    
	    d_B.width  = m_B.getPatchY();
	    d_B.height = m_B.getPatchX(); 
	    d_B.stride = m_B.getX();
	    d_B.elements = m_B.getDataPatchConst();


	#ifdef CUBLAS
		// In case of patch usage don,t use cublas
		if(m_A.isPatched() || m_B.isPatched()){
            assert(d_A.width==d_B.height);
            outMatrix.Reset(d_A.height,d_B.width);

            d_C.width  = outMatrix.getY();
            d_C.height = d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();
		    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		    dim3 dimGrid((d_B.width+BLOCK_SIZE-1) / dimBlock.x, 
			     (d_A.height+BLOCK_SIZE-1)/ dimBlock.y);
		    MatMulKernel<<<dimGrid, dimBlock, 0, stream>>>(
			    d_A, 
			    d_B, 
			    d_C, 
			    d_A.height, 
			    d_A.width,
			    d_B.height, 
			    d_B.width,
			    d_C.height, 
			    d_C.width);
		}
	    else{
		    const float m_One = 1.0f;
		    const float m_Zero = 0.0f;


            int x = !m_A.isTrans() ? m_A.getX() : m_A.getY();
            int y = !m_B.isTrans() ? m_B.getY() : m_B.getX();
            int kA = !m_A.isTrans() ? m_A.getY() : m_A.getX();
            int kB = !m_B.isTrans() ? m_B.getX() : m_B.getY();

            assert(kA == kB);

            outMatrix.Reset(x, y);

            cublasSgemm(
                inHandle, 
                !m_A.isTrans() ? CUBLAS_OP_N : CUBLAS_OP_T, 
                !m_B.isTrans() ? CUBLAS_OP_N : CUBLAS_OP_T,
                x, y, kA,
                &m_One, m_A.getDataConst(), 
                m_A.getX(), 
                m_B.getDataConst(), 
                m_B.getX(), 
                &m_Zero, 
                outMatrix.getDataConst(), 
                x);
            /*
		    cublasSgemm(inHandle, 
			    CUBLAS_OP_N, 
			    CUBLAS_OP_N, 
			    d_A.height, 
			    d_B.width, 
			    d_A.width,
			    &m_One, 
			    d_A.elements, 
			    d_A.height, 
			    d_B.elements, 
			    d_B.height, 
			    &m_Zero, 
			    d_C.elements, 
			    d_C.height);
            */
		}
	#else
		// Invoke kernel
        assert(d_A.width==d_B.height);
        outMatrix.Reset(d_A.height,d_B.width);

        d_C.width  = outMatrix.getY();
        d_C.height = d_C.stride = outMatrix.getX();
        d_C.elements = outMatrix.getData();
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((d_B.width+BLOCK_SIZE-1) / dimBlock.x, 
			     (d_A.height+BLOCK_SIZE-1)/ dimBlock.y);
        MatMulKernel<<<dimGrid, dimBlock, 0, stream>>>(d_A,
                d_B, 
                d_C, 
                d_A.height, 
                d_A.width,
                d_B.height, 
                d_B.width,
                d_C.height, 
                d_C.width);
        
    #endif

}

// Get a matrix element
__device__ float GetElement(const MatrixPointer A, int row, int col) {
    return A.elements[col * A.stride + row];
}

// Set a matrix element
__device__ void SetElement(MatrixPointer A, int row, int col, float value) {
    A.elements[col * A.stride + row] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ MatrixPointer GetSubMatrix(MatrixPointer A, int row, int col) {
    MatrixPointer Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * col + BLOCK_SIZE * row];
    return Asub;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(
        MatrixPointer A, 
        MatrixPointer B, 
        MatrixPointer C , 
        int ARows, 
        int ACols, 
        int BRows, 
        int BCols, 
        int CRows, 
        int CCols){

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    MatrixPointer Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0.0;

    for (int m = 0; m < ((A.width+BLOCK_SIZE-1) / BLOCK_SIZE); m++) {
        // Get sub-matrix Asub of A
        MatrixPointer Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        MatrixPointer Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // Switch rows and cols
        if((m*BLOCK_SIZE+row<ACols) && (blockRow*BLOCK_SIZE+col<ARows))
            As[col][row] = GetElement(Asub, col, row);
        else 
            As[col][row] = 0;

        if((m*BLOCK_SIZE+col<BRows)  && (blockCol*BLOCK_SIZE+row<BCols))
            Bs[col][row] = GetElement(Bsub, col, row);
        else
            Bs[col][row] = 0;
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    if(((blockRow*BLOCK_SIZE+row)<CRows) && ((blockCol*BLOCK_SIZE+col)<CCols))
        SetElement(Csub, row, col, Cvalue);
}


/* Operators of MatrixAsync */
template<typename Matrix> 
OperationMatrixMultiplyAsync<Matrix> MatrixAsync<Matrix>::operator*(
        const MatrixAsync<Matrix>& inB) const{
    return OperationMatrixMultiplyAsync<Matrix>(*this, inB);
}

template<typename Matrix> 
void MatrixAsync<Matrix>::operator=(const OperationAsyncGpu<Matrix>& inOperation){
        inOperation.Execute(*this,handle,stream);   
}

template<typename Matrix>
MatrixAsync<Matrix>& MatrixAsync<Matrix>::operator=(const MatrixCpu &inMatrix){
    matrix = inMatrix; 
    return *this;
}

template<typename Matrix>
MatrixAsync<Matrix>& MatrixAsync<Matrix>::operator=(const MatrixAsync<Matrix> &inMatrix){
    matrix = inMatrix.getMatrixConst(); 
    return *this;
}

/* MatrixAsyncOperations                                                      * 
 ******************************************************************************/
__global__ void MatNonlinDleta(MatrixPointer Delta, MatrixPointer Y, MatrixPointer Result,  int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size){
        Result.elements[tid] = (1 - Y.elements[tid]) * Y.elements[tid] * Delta.elements[tid];
    }        
}

__global__ void MatMulConstKernel(MatrixPointer A, MatrixPointer B, float constant, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<size){
        B.elements[tid] = constant * A.elements[tid];
    }
}


__global__ void ActivKernel(MatrixPointer A, MatrixPointer B, int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid<size){
        B.elements[tid] = (float)1.0/(1 + powf((float)M_E,-A.elements[tid]));
    }
}

__global__ void MatSubKernel(MatrixPointer inA, MatrixPointer inB, MatrixPointer outC, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size){
        outC.elements[tid] = inA.elements[tid] - inB.elements[tid];
    }
}

__global__ void MatAddKernel(MatrixPointer A, MatrixPointer B, MatrixPointer C, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size){
        C.elements[tid] = A.elements[tid] + B.elements[tid];
    }

}

__global__ void WeightKernel(MatrixPointer W, MatrixPointer Eps, MatrixPointer Result, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<size){
        Result.elements[tid] = W.elements[tid] * Eps.elements[tid];
    }
}


__global__ void MatReduceKernel(MatrixPointer A, MatrixPointer B){
    /* TODO reimplement*/
    int basePos =  blockIdx.x*blockDim.x + threadIdx.x;
    if(basePos>=A.width)
        return;
    float sum  = 0.0;
    for(int i = 0; i<A.height; i++)
        sum += A.elements[basePos*A.stride + i];
    B.elements[basePos*B.stride] = sum;
}


__global__ void EpsKernel(MatrixPointer lastDir, MatrixPointer actDir, MatrixPointer lSpeed, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid<size){
        bool goodDir = (lastDir.elements[tid]*actDir.elements[tid]) >= 0;
        lSpeed.elements[tid] *= goodDir ? 1.1f : 0.5f;
        if(lSpeed.elements[tid] < (float)0.0000000001){
            lSpeed.elements[tid] = (float)0.0000000001;
        }
        else if(lSpeed.elements[tid] > (float)0.00001){
            lSpeed.elements[tid] = (float)0.00001;
        }
    }
}

template<typename Matrix> void OperationMatrixAsync<Matrix>::Execute(
         MatrixAsync<Matrix> &outMatrix, 
         cublasHandle_t inHandle,
         cudaStream_t stream) const{

    static int ThreadsPerBlock;
    dim3 dimBlock(1,1);
    dim3 dimGrid(1,1);
    int num;
    int blocks;

    MatrixPointer d_A, d_B, d_C; 
    d_A.width  = m_A.getPatchY();
    d_A.height = m_A.getPatchX();
    d_A.stride = m_A.getX();
    d_A.elements = m_A.getDataPatchConst();
    d_B.width  = m_B.getPatchY();
    d_B.height = m_B.getPatchX(); 
    d_B.stride = m_B.getX();
    d_B.elements = m_B.getDataPatchConst();

    switch(operationType){
        case SubErrorCountOperation:    
            assert(d_A.width==d_B.width);
    	    assert(d_A.height==d_B.height);

            if((outMatrix.getPatchX() != d_A.height) ||
               (outMatrix.getPatchY() != d_A.width))
                    outMatrix.Reset(d_A.height,d_A.width);
            d_C.width  = outMatrix.getPatchY();
            d_C.height = outMatrix.getPatchX();
            d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();

            ThreadsPerBlock = 512;
            num = m_A.getX()*m_A.getY();
            blocks = (num - 1) / ThreadsPerBlock + 1;
            MatSubKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(d_A, d_B, d_C, num);
            break;
        case AddOperation:    
            assert(d_A.width==d_B.width);
            assert(d_A.height==d_B.height);

            if((outMatrix.getPatchX() != d_A.height) ||
               (outMatrix.getPatchY() != d_A.width))
                    outMatrix.Reset(d_A.height,d_A.width);
            d_C.width  = outMatrix.getPatchY();
            d_C.height = outMatrix.getPatchX();
            d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();

            ThreadsPerBlock = 512;
            num = m_A.getX()*m_A.getY();
            blocks = (num - 1) / ThreadsPerBlock + 1;
            MatAddKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(d_A, d_B, d_C, num);
            break;
        case ReduceOperation: 
            if(outMatrix.getPatchY() != d_A.width)
                outMatrix.Reset(1,d_A.width);
            d_C.width  = outMatrix.getPatchY();
            d_C.height = outMatrix.getPatchX();
            d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();

            dimBlock.x = BLOCK_SIZE;
            dimBlock.y = 1;
            dimGrid.x = (d_C.width+BLOCK_SIZE-1) / dimBlock.x;
            dimGrid.y = 1;
            MatReduceKernel<<<dimGrid,dimBlock,0, stream>>>(d_A, d_C);
            break;
        case ActivateOperation:
            if((outMatrix.getPatchY() != d_A.width) && (outMatrix.getPatchX() != d_A.height))
                outMatrix.Reset(d_A.height,d_A.width);
            d_C.width  = outMatrix.getPatchY();
            d_C.height = outMatrix.getPatchX();
            d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();

            ThreadsPerBlock = 512;
            num = m_A.getX()*m_A.getY();
            blocks = (num - 1) / ThreadsPerBlock + 1;

            ActivKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(d_A, d_C, num);
            break;
        case ConstMultOperation:
            if((outMatrix.getPatchY() != d_A.width) && (outMatrix.getPatchX() != d_A.height))
                outMatrix.Reset(d_A.height,d_A.width);
            d_C.width  = outMatrix.getPatchY();
            d_C.height = outMatrix.getPatchX();
            d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();

            ThreadsPerBlock = 512;
            num = m_A.getX()*m_A.getY();
            blocks = (num - 1) / ThreadsPerBlock + 1;

            MatMulConstKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(d_A, d_C, constant, num);
            break;
        case DeltaNonlinOperation:
            if((outMatrix.getPatchY() != d_A.width) || (outMatrix.getPatchX() != d_A.height))
                outMatrix.Reset(d_A.height,d_A.width);
            assert(d_A.height == d_B.height);
            assert(d_A.width == d_B.width);

            d_C.width  = outMatrix.getPatchY();
            d_C.height = outMatrix.getPatchX();
            d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();

            ThreadsPerBlock = 512;
            num = m_A.getX()*m_A.getY();
            blocks = (num - 1) / ThreadsPerBlock + 1;

            MatNonlinDleta<<<blocks, ThreadsPerBlock, 0, stream>>>(d_A, d_B, d_C, num);
            break;
        case EpsOperation:
            if((outMatrix.getPatchY() != d_A.width) || (outMatrix.getPatchX() != d_A.height))
                outMatrix.Reset(d_A.height,d_A.width);
            assert(d_A.height == d_B.height);
            assert(d_A.width == d_B.width);

            d_C.width  = outMatrix.getPatchY();
            d_C.height = outMatrix.getPatchX();
            d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();

            ThreadsPerBlock = 512;
            num = m_A.getX()*m_A.getY();
            blocks = (num - 1) / ThreadsPerBlock + 1;

            EpsKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(d_A,d_B,d_C,num);
            break;
        case WeightOperation:
            if((outMatrix.getPatchY() != d_A.width) || (outMatrix.getPatchX() != d_A.height))
                outMatrix.Reset(d_A.height,d_A.width);
            assert(d_A.height == d_B.height);
            assert(d_A.width == d_B.width);

            d_C.width  = outMatrix.getPatchY();
            d_C.height = outMatrix.getPatchX();
            d_C.stride = outMatrix.getX();
            d_C.elements = outMatrix.getData();

            ThreadsPerBlock = 512;
            num = m_A.getX()*m_A.getY();
            blocks = (num - 1) / ThreadsPerBlock + 1;
            WeightKernel<<<blocks, ThreadsPerBlock, 0, stream>>>(d_A, d_B, d_C, num);
            break;
        default:
            break;
    }
}
     
template<typename Matrix> OperationMatrixAsync<Matrix>::OperationMatrixAsync(
        const MatrixAsync<Matrix>& inA, 
        const MatrixAsync<Matrix>& inB, 
        Operation operationType):m_A(inA), m_B(inB), operationType(operationType), constant(0){}

template<typename Matrix> OperationMatrixAsync<Matrix>::OperationMatrixAsync(
                const MatrixAsync<Matrix>& inA, 
                const float constant, 
                Operation operationType): m_A(inA), m_B(inA), operationType(operationType), constant(constant){}

template<typename Matrix>
OperationMatrixAsync<Matrix> MatrixAsync<Matrix>::operator-(
        const MatrixAsync<Matrix>& inB) const{
    return OperationMatrixAsync<Matrix>(*this, inB, SubErrorCountOperation);
}

template<typename Matrix>
OperationMatrixAsync<Matrix> MatrixAsync<Matrix>::operator+(
        const MatrixAsync<Matrix>& inB) const{
    return OperationMatrixAsync<Matrix>(*this, inB, AddOperation);
}

template<typename Matrix>
OperationMatrixAsync<Matrix> MatrixAsync<Matrix>::Reduce(void) const{
    return OperationMatrixAsync<Matrix>(*this, *this, ReduceOperation);
}

template<typename Matrix>
OperationMatrixAsync<Matrix> MatrixAsync<Matrix>::Activate(void) const{
    return OperationMatrixAsync<Matrix>(*this, *this, ActivateOperation);
}


template<typename Matrix>
OperationMatrixAsync<Matrix> MatrixAsync<Matrix>::operator*(const float constant) const{
    return OperationMatrixAsync<Matrix>(*this, constant, ConstMultOperation);
}

template<typename Matrix> void
MatrixAsync<Matrix>::ReduceInSitu(void){
    OperationMatrixAsync<Matrix> op(*this, *this, ReduceOperation);
    op.Execute(*this, handle, stream);

}

template<typename Matrix> void
MatrixAsync<Matrix>::ActivateInSitu(void){
    OperationMatrixAsync<Matrix> op(*this, *this, ActivateOperation);
    op.Execute(*this, handle, stream);
}

template<typename Matrix> 
OperationMatrixAsync<Matrix> MatrixAsync<Matrix>::DeltaNonlinCount(const MatrixAsync<Matrix>& inB) const{
    return OperationMatrixAsync<Matrix>(*this, inB, DeltaNonlinOperation);
}

template<typename Matrix> void 
MatrixAsync<Matrix>::DeltaNonlinCountInSitu(const MatrixAsync<Matrix>& inB){
    OperationMatrixAsync<Matrix> op(*this, inB, DeltaNonlinOperation);
    op.Execute(*this, handle, stream);
}

template<typename Matrix> void 
MatrixAsync<Matrix>::EpsRefreshInSitu(const MatrixAsync<Matrix>& deltaOld, const MatrixAsync<Matrix>& deltaNew){
    OperationMatrixAsync<Matrix> op(deltaOld, deltaNew, EpsOperation);
    op.Execute(*this, handle, stream);
}


template<typename Matrix> void 
MatrixAsync<Matrix>::WeightRefreshInSitu(const MatrixAsync<Matrix>& eps){
    OperationMatrixAsync<Matrix> op(*this, eps,  WeightOperation);
    op.Execute(*this, handle, stream);
}


void msgC(char * inMsg, const MatrixCpu &x){
    int n = x.getX()*x.getY();
    cout<<string(inMsg)<<x.getX()<<" "<<x.getY()<<endl;
    
    for(int i=0; i < min(5,x.getX()); i++){
        for(int e=0 ; e < min(5,x.getY()); e++)
            cout<<x.getDataConst()[e*x.getX()+i]<<" ";
        cout<<endl;
    }
    cout<<"-------------------------------------"<<endl;
    for(int i=max(x.getX()-5,0); i < x.getX(); i++){
        cout<<"\t";
        for(int e=max(x.getY()-5,0) ; e < x.getY(); e++)
            cout<<x.getDataConst()[e*x.getX()+i]<<" ";
        cout<<endl;
    }
    cout<<endl;
}
    
void msgG(char * inMsg, const MatrixAsync<MatrixGpu> &inM){
    if(!inM.isPatched()){
        MatrixCpu x = inM.getMatrixConst();
        msgC(inMsg, x);
    }
}

void initGTrTe( MatrixAsync<MatrixGpu>& outMGpuInTr, MatrixAsync<MatrixGpu>& outMGpuInTe, string filename){
    MatrixCpu mCpu;
    ifstream ifx;

    ifx.clear();
    ifx.open(filename.c_str());
    mCpu.Load(ifx, false);
    ifx.close();

    int rows = mCpu.getX();
    int cols = mCpu.getY();
    int fract = rows - rows/5;

    outMGpuInTr = mCpu.SubMatrix(0, 0, fract, cols);
    outMGpuInTe  = mCpu.SubMatrix(fract, 0, rows, cols);
}

void initG( MatrixAsync<MatrixGpu>& outMGpu, string filename){
    MatrixCpu mCpu;
    ifstream ifx;

    ifx.clear();
    ifx.open(filename.c_str());
    mCpu.Load(ifx, false);
    ifx.close();

    outMGpu = mCpu;
}

float computeError(MatrixGpu &inR, MatrixGpu &inOut){
    MatrixGpu r, r2, r3;
    r2 = inR - inOut;
    r2 ^= 2.0f;
    r3 = r2.AbsSum();
    r3 *= 1.0f / inOut.getX();
    MatrixCpu rr = r3;
    return rr.getDataConst()[0];
}


void saveMatrix(MatrixAsync<MatrixGpu> &inM, const char* filename){
    ofstream f(filename);
    MatrixCpu m = inM.getMatrix();
    m.Save(f);
    f.close();
}

static volatile int interrupt = 0;
void intHandlerFce(int signal){
    if(signal == SIGINT)
        interrupt = 1;
}



void learn(char* argv[]){

    const int brainSize = 3;
    Error<3> err(2000);
    string w0Out = argv[4];
    w0Out += "0";
    string w1Out = argv[4];
    w1Out += "1";
    string w2Out = argv[4];
    w2Out += "2";

    struct sigaction intHandler;
    intHandler.sa_handler = intHandlerFce;
    sigemptyset(&intHandler.sa_mask);
    intHandler.sa_flags = 0;
    sigaction(SIGINT, &intHandler, NULL);

    MatrixAsync<MatrixGpu> deltas[3];
    MatrixAsync<MatrixGpu> innerLayer[4];
    MatrixAsync<MatrixGpu> innerLayerTest[4];
    MatrixAsync<MatrixGpu> desire;
    MatrixAsync<MatrixGpu> desireTest;

    initGTrTe(innerLayer[0], innerLayerTest[0], argv[2]);
    initGTrTe(desire, desireTest, argv[3]);


    innerLayer[1].Reset(innerLayer[0].getX(), 1000);
    innerLayer[2].Reset(innerLayer[0].getX(), 500);
    innerLayer[3].Reset(innerLayer[0].getX(), desire.getY());
    
    innerLayerTest[1].Reset(innerLayerTest[0].getX(), 1000);
    innerLayerTest[2].Reset(innerLayerTest[0].getX(), 500);
    innerLayerTest[3].Reset(innerLayerTest[0].getX(), desire.getY());

    /* Weights initialization */
    MatrixAsync<MatrixGpu> weights[3];
    MatrixAsync<MatrixGpu> weightsSave[3];
    //weights[0].Reset(4096,1000);
    //weights[1].Reset(1000,500);
    //weights[2].Reset(500,300);
    // Randomized   +-3/sqrt(N)
    if(innerLayer[0].getY() == 1000)
        initG(weights[0], "initW/wSmall0");
    else if(innerLayer[0].getY() == 4096)
        initG(weights[0], "initW/wBig0");
    else if(innerLayer[0].getY() == 5096)
        initG(weights[0], "initW/wSmallBig0");
    initG(weights[1], "initW/w1");
    initG(weights[2], "initW/w2");


    MatrixAsync<MatrixGpu> weightsDelta[3];
    weightsDelta[0].Reset(innerLayer[0].getY(),1000);
    weightsDelta[1].Reset(1000,500);
    weightsDelta[2].Reset(500,desire.getY());

    MatrixAsync<MatrixGpu> weightsDeltaSave[3];
    weightsDeltaSave[0].Reset(innerLayer[0].getY(),1000);
    weightsDeltaSave[1].Reset(1000,500);
    weightsDeltaSave[2].Reset(500,desire.getY());



    MatrixAsync<MatrixGpu> eps[3];
    eps[0].Reset(weights[0].getX(),1000);
    eps[1].Reset(1000,500);
    eps[2].Reset(500,desire.getY()); 
    for(int id = 0; id<brainSize; id++)
        eps[id].getMatrix() = 0.0000001;
    
    cout<<"Data loaded successfully."<<endl;

    for(int iter=0; iter<200000; iter++){
        /*                           ERRA CHECK                            *
        /**********************************************************************/ 
        for(int layerId = 0; layerId < brainSize; layerId++){
            innerLayerTest[layerId+1] = innerLayerTest[layerId] * weights[layerId];
            cudaDeviceSynchronize();
            if((layerId+1)<brainSize)
                innerLayerTest[layerId+1].ActivateInSitu();
            cudaDeviceSynchronize();
        }
        err.next(computeError(innerLayerTest[brainSize].getMatrix(), desireTest.getMatrix()));     
        if(err.loadOldW()){
            for(int layerId = 0; layerId < brainSize; layerId++){
                weightsSave[layerId] = weights[layerId] + weightsDelta[layerId]; 
            }
            err.loadW(weightsSave);
        }
        if(err.loadNewW()){
            err.loadW(weights);
        }
        if(err.finished()) break;
        /*                            FORWARD PASS                            *
        /**********************************************************************/
        for(int layerId = 0; layerId < brainSize; layerId++){
            innerLayer[layerId+1] = innerLayer[layerId] * weights[layerId];
            cudaDeviceSynchronize();
            if((layerId+1)<brainSize)
                innerLayer[layerId+1].ActivateInSitu();
            cudaDeviceSynchronize();
        }
        /*                            BACKWARD PASS                           *
        /**********************************************************************/
        /*                            delta recount                           *
        /**********************************************************************/ 
        deltas[brainSize-1] = innerLayer[brainSize] - desire;
        cudaDeviceSynchronize();
        #ifdef ERR  
            if(iter%100 == 0){
                for(int layerId = 0; layerId < brainSize; layerId++){
                    innerLayerTest[layerId+1] = innerLayerTest[layerId] * weights[layerId];
                    cudaDeviceSynchronize();
                    if((layerId+1)<brainSize)
                        innerLayerTest[layerId+1].ActivateInSitu();
                    cudaDeviceSynchronize();
                }     
                cout<<"Error ("<<iter<<") "<<computeError(innerLayer[brainSize].getMatrix(),
                desire.getMatrix())<<" "<<computeError(innerLayerTest[brainSize].getMatrix(),
                desireTest.getMatrix())<<std::endl;
            }
        #endif   
        cudaDeviceSynchronize();
        for(int layerId = brainSize-1; layerId >= 1; layerId--){
            weights[layerId].Transpose();
            deltas[layerId-1] = deltas[layerId]*weights[layerId];
            cudaDeviceSynchronize();
            deltas[layerId-1].DeltaNonlinCountInSitu(innerLayer[layerId]);
            cudaDeviceSynchronize();
            weights[layerId].Transpose();
        }
        
        /*                        weight delta recount                        *
        /**********************************************************************/
        for(int layerId = 0; layerId < brainSize; layerId++){
            innerLayer[layerId].Transpose();
            weightsDeltaSave[layerId] = weightsDelta[layerId];
            weightsDelta[layerId] = innerLayer[layerId]*deltas[layerId];
        }
        cudaDeviceSynchronize();
        if(iter!=0){
            for(int layerId = 0; layerId < brainSize; layerId++){
                eps[layerId].EpsRefreshInSitu(weightsDeltaSave[layerId], weightsDelta[layerId]);
            }
        }
        cudaDeviceSynchronize();
        for(int layerId = 0; layerId < brainSize; layerId++){
            //weightsDelta[layerId] = weightsDelta[layerId] * 0.0000001;
            weightsDelta[layerId].WeightRefreshInSitu(eps[layerId]);
        }    
        for(int layerId = 0; layerId < brainSize; layerId++)
            innerLayer[layerId].Transpose();
        cudaDeviceSynchronize();
        /*                        new weights count                           *
        /**********************************************************************/
        for(int layerId = 0; layerId < brainSize; layerId++){
            weights[layerId] = weights[layerId] - weightsDelta[layerId]; 
        }
        cudaDeviceSynchronize(); 
        if(interrupt==1) break; 
    }
    saveMatrix(err.w[0], w0Out.c_str());
    saveMatrix(err.w[1], w1Out.c_str());
    saveMatrix(err.w[2], w2Out.c_str());
    cout<<err.oldValue<<endl;    
}


void eval(char* argv[]){
    const int brainSize = 3;
    string w0In = argv[4];
    w0In += "0";
    string w1In = argv[4];
    w1In += "1";
    string w2In = argv[4];
    w2In += "2";

    /* Weights initialization */
    MatrixAsync<MatrixGpu> weights[3];
    initG(weights[0], w0In);
    initG(weights[1], w1In);
    initG(weights[2], w2In);

    MatrixAsync<MatrixGpu> innerLayer[4];

    initG(innerLayer[0], argv[2]);
    innerLayer[1].Reset(innerLayer[0].getX(), weights[0].getY());
    innerLayer[2].Reset(innerLayer[0].getX(), weights[1].getY());
    innerLayer[3].Reset(innerLayer[0].getX(), weights[2].getY());
    
    cout<<"Data loaded successfully."<<endl;

    for(int layerId = 0; layerId < brainSize; layerId++){
        innerLayer[layerId+1] = innerLayer[layerId] * weights[layerId];
        cudaDeviceSynchronize();
        if((layerId+1)<brainSize)
            innerLayer[layerId+1].ActivateInSitu();
        cudaDeviceSynchronize();
    }
    saveMatrix(innerLayer[3], argv[3]);
}

int main(int argc, char** argv){

    if(argc!=5){
        cout<<argv[0]<<"-[l e]"<<"input "<<"output "<<"weight"<<endl;
        return 0;
    }
    if(!strcmp(argv[1], "-l"))
        learn(argv);
    else if(!strcmp(argv[1], "-e"))
        eval(argv);
    else
        cout<<"Uknown params >"<<argv[1]<<"<"<<endl;
    return 0;
}

 