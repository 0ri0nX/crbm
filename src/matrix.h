#ifndef MATRIX_H
#define MATRIX_H

#include <cublas_v2.h>
#include <curand.h>
#include <limits>
#include <float.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <stdint.h>
#include <cassert>
#include <cstdio>
#include <stdexcept>

//for memory map
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "matrixCpu.h"

void msgG(const char * inMsg, const YAMATH::MatrixGpu &inM);

namespace YAMATH
{
    typedef unsigned long t_index;

//#define DEBUG_MATRIX_CLASS
//#define DEBUG_ALLOCATION
//#define DEBUG_CLASS

//#define debugMatrix(m) { std::cout << "matrix at " << __LINE__ << ": " << (m).getX() << " x " << (m).getY() << ((m).isTrans() ? "T" : "") <<  std::endl;}
#define debugMatrix(m) 

#ifdef DEBUG_CLASS
    #define DEB_CONSTRUCTOR(a) { a##Number += 1; std::cout << "c" << a##Id << "(" << a##Number << ")"; }
    #define DEB_DESTRUCTOR(a) { a##Number -= 1; std::cout << "d" << a##Id << "(" << a##Number << ")"; }
    #define DEB_INIT(a, id) int a##Number = 0; char *a##Id = id;
#else
    #define DEB_CONSTRUCTOR(a)
    #define DEB_DESTRUCTOR(a) 
    #define DEB_INIT(a, id)
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      assert(!abort);
      if (abort) exit(code);
   }
}

#ifdef DEBUG_ALLOCATION
    int xxx = 0;
#endif

    template <typename T>
    inline T* allocateGpu(t_index inNum)
    {
#ifdef DEBUG_ALLOCATION
        ++xxx;
        cout << "a(" << xxx << ": " << inNum << " x " << sizeof(T) << ")";
#endif
        T *ptr = NULL;
        gpuErrchk(cudaMalloc((void**) &ptr, inNum*sizeof(T)));
        assert(ptr != NULL);
        return ptr;
    }

    template <typename T>
    inline void deallocateGpu(T *inArray)
    {
#ifdef DEBUG_ALLOCATION
        --xxx;
        cout << "d";
#endif
        cudaFree(inArray);
    }

    template <typename T>
    class GpuData
    {
        struct Holder
        {
            Holder(t_index inNum) :
                m_Counter(1), m_Data(NULL)
            {
                //cudaMalloc((void**) &m_Data, inNum*sizeof(T));
                //assert(m_Data != NULL);
                m_Data = allocateGpu<T>(inNum);
            }

            ~Holder(void)
            {
                assert(m_Counter == 0);
                //cudaFree(m_Data);
                deallocateGpu<T>(m_Data);
            }
    
            int m_Counter;
            float *m_Data;
        };

        public:
            GpuData(t_index inNum = 0)
            {
                //assert(inNum >= 0);

                if(inNum != 0)
                {
                    m_Holder = new Holder(inNum);
                }
                else
                {
                    m_Holder = NULL;
                }
            }

            GpuData(const GpuData<T> &inVal)
            {
                m_Holder = inVal.m_Holder;

                if(m_Holder != NULL)
                {
                    m_Holder->m_Counter += 1;
                }
            }

            ~GpuData(void)
            {
                Dec();
            }

            T* raw(void) const
            {
                if (m_Holder != NULL)
                {
                    return m_Holder->m_Data;
                }
                else
                {
                    return NULL;
                }
            }

            GpuData<T> &operator=(t_index inNum)
            {
                Dec();

                if(inNum != 0)
                {
                    m_Holder = new Holder(inNum);
                }
                return *this;
            }
            GpuData<T> &operator=(const GpuData<T> &inVal)
            {
                if(inVal.m_Holder == NULL)
                {
                    Dec();
                }
                else if(inVal.m_Holder != NULL && m_Holder != inVal.m_Holder)
                {
                    Dec();

                    m_Holder = inVal.m_Holder;
                    m_Holder->m_Counter += 1;
                }

                return *this;
            }

            void Reset(t_index inNum = 0)
            {
                Dec();
                if(inNum != 0)
                {
                    m_Holder = new Holder(inNum);
                }
            }

        private:

            void Dec(void)
            {
                if(m_Holder != NULL)
                {
                    m_Holder->m_Counter -= 1;
                    if(m_Holder->m_Counter == 0)
                    {
                        delete m_Holder;
                    }
    
                    m_Holder = NULL;
                }
            }
    
            Holder *m_Holder;

            template <typename TT>
            friend void swapData(GpuData<TT> &inA, GpuData<TT> &inB);
    };

    template <typename T>
        void swapData(GpuData<T> &inA, GpuData<T> &inB)
        {
            typename GpuData<T>::Holder *tmp = inA.m_Holder;
            inA.m_Holder = inB.m_Holder;
            inB.m_Holder = tmp;
        }

    enum EFunctionElementwiseBinary
    {
        EFEB_Plus,
        EFEB_Minus,
        EFEB_Multiply,
        EFEB_Divide,
    };

    enum EFunctionElementwise
    {
        EFE_Square,
        EFE_Sqrt,
        EFE_Pow,
        EFE_ScalarMultiply,
        EFE_Fill,
        EFE_Less,
        EFE_LessOrEqual,
        EFE_Equal,
        EFE_GreaterOrEqual,
        EFE_Greater,
        EFE_NotEqual,
        EFE_PlusScalar,
        EFE_MinusScalar,
        EFE_MultiplyScalar,
        EFE_DivideScalar,
        EFE_InverseAndMultiply,
        EFE_Sigmoid,
        EFE_Minimally,
        EFE_Maximally,
    };

    enum EAggregate
    {
        EA_AbsSum,
        EA_AbsMin,
        EA_AbsMax,
        EA_Sum,
    };

    enum EFunctionBinaryAssociative
    {
        EFB_Plus,
        EFB_Multiply,
        EFB_Max,
        EFB_Min,
    };

    //forwards
    class MatrixCpu;
    class MatrixGpu;
    class OperationGpu;
    class OperationMatrixMultiply;
    class OperationMatrixAdd;
    class OperationMatrixSubstract;
    class OperationMatrixApplyElementwise;
    class OperationMatrixAggregate;
    class OperationBinaryAssociative;
    class OperationMatrixElementwiseBinary;
    class MatrixCpu;

    DEB_INIT(MatrixGpu, "a");
    DEB_INIT(OperationGpu, "b");
    DEB_INIT(OperationMatrixMultiply, "c");
    DEB_INIT(OperationMatrixAdd, "d");
    DEB_INIT(OperationMatrixSubstract, "e");
    DEB_INIT(OperationMatrixApplyElementwise, "f");
    DEB_INIT(OperationMatrixAggregate, "g");
    DEB_INIT(OperationBinaryAssociative, "h");
    DEB_INIT(OperationMatrixElementwiseBinary, "i");


    class MatrixGpu
    {
        public:
            MatrixGpu(void)
                : m_X(1), m_Y(1), m_Data(0), m_Transposed(false)
            {
                m_Data.Reset(1);
#ifdef DEBUG_MATRIX_CLASS
                m_Allocations += 1;
                cout << "m" << m_Allocations;
#endif
            }


            MatrixGpu(t_index x, t_index y, bool inTransposed = false)
                : m_X(x), m_Y(y), m_Data(0), m_Transposed(inTransposed)
            {
                m_Data.Reset(m_X*m_Y);
#ifdef DEBUG_MATRIX_CLASS
                m_Allocations += 1;
                cout << "m" << m_Allocations;
#endif
            }

//column-first order - ld is leading dimension size - #rows
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

            MatrixGpu(const MatrixGpu &inMatrix, bool inShallowCopy = false);
            MatrixGpu(const MatrixCpu &inMatrix);
            MatrixGpu(const OperationGpu &inOperation);

            ~MatrixGpu(void)
            {
#ifdef DEBUG_MATRIX_CLASS
                m_Allocations -= 1;
                cout << "d" << m_Allocations;
#endif
            }

            t_index getX(void) const { return m_X; }
            t_index getY(void) const { return m_Y; }
            float* getDataConst(void) const { return m_Data.raw(); }
            float* getData(void) { return m_Data.raw(); }

            bool isTrans(void) const { return m_Transposed; }

            void Transpose(void) { m_Transposed = !m_Transposed; }

            MatrixGpu T(void) const
            {
                //std::cout << "before T(): " << getX() << " x " << getY() << std::endl;
                MatrixGpu m(*this, true);
                m.Transpose();
                //std::cout << "after T(): " << m.getX() << " x " << m.getY() << std::endl;

                return m;
            }

            void MakeHardCopy(void);

            void Sample(t_index inRowsNum, MatrixGpu &outSample) const;

            void Reset(t_index inX, t_index inY, bool inTransposed = false)
            {
                if(m_X*m_Y != inX*inY || m_ShallowCopy)
                {
                    m_Data = 0;

                    Init(inX, inY, inTransposed);
                }
                else
                {
                    m_Transposed = inTransposed;
                    m_ShallowCopy = false;
                    m_X = inX;
                    m_Y = inY;
                }
            }

            void Reshape(t_index inX, t_index inY)
            {
                assert(getX()*getY() == inX*inY);

                m_X = inX;
                m_Y = inY;
            }

            void RandUniform(unsigned long long inSeed = 0)//uniform randomness (0.0f .. 1.0f]
            {
                // Create a pseudo-random number generator
                curandGenerator_t prng;
                curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

                // Set the seed for the random number generator using the system clock
                curandSetPseudoRandomGeneratorSeed(prng, inSeed != 0 ? inSeed : (unsigned long long) clock());

                // Fill the array with random numbers on the device
                curandGenerateUniform(prng, m_Data.raw(), m_X*m_Y);

                //destroy generator
                curandDestroyGenerator(prng);
            }

            void RandNormal(float inMean, float inStdDev, unsigned long long inSeed = 0)//normal randomess, mean 0.0f standard deviation 1.0f
            {
                // Create a pseudo-random number generator
                curandGenerator_t prng;
                curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

                // Set the seed for the random number generator using the system clock
                curandSetPseudoRandomGeneratorSeed(prng, inSeed != 0 ? inSeed : (unsigned long long) clock());

                // Fill the array with random numbers on the device
                curandGenerateNormal(prng, m_Data.raw(), m_X*m_Y, inMean, inStdDev);

                //destroy generator
                curandDestroyGenerator(prng);
            }

            MatrixGpu &operator=(const OperationGpu &inOperation);
            MatrixGpu &operator=(float inFill);
            MatrixGpu &operator=(const MatrixCpu& inMatrix);
            MatrixGpu &operator=(const MatrixGpu& inMatrix);

            //comparison with one float returns matrix where each element is equal 1.0 if comparison is true otherwise 0.0
            OperationMatrixApplyElementwise operator<(float  inVal) const;
            OperationMatrixApplyElementwise operator<=(float inVal) const;
            OperationMatrixApplyElementwise operator==(float inVal) const;
            OperationMatrixApplyElementwise operator>=(float inVal) const;
            OperationMatrixApplyElementwise operator>(float  inVal) const;
            OperationMatrixApplyElementwise operator!=(float inVal) const;

            //elementwise binary matrix operators: + - * /
            OperationMatrixElementwiseBinary operator+(const MatrixGpu &inB) const;
            OperationMatrixElementwiseBinary operator-(const MatrixGpu &inB) const;
            OperationMatrixElementwiseBinary operator*(const MatrixGpu &inB) const;//elementwise multiplication!
            OperationMatrixElementwiseBinary operator/(const MatrixGpu &inB) const;

            MatrixGpu &operator+=(const MatrixGpu &inB);
            MatrixGpu &operator-=(const MatrixGpu &inB);
            MatrixGpu &operator*=(const MatrixGpu &inB);//elementwise multiplication!
            MatrixGpu &operator/=(const MatrixGpu &inB);

            //aggregation functions over all elements of a matrix
            OperationBinaryAssociative Sum(void) const;
            OperationBinaryAssociative Min(void) const;
            OperationBinaryAssociative Max(void) const;
            OperationBinaryAssociative Product(void) const;
            //aggregation over absolute values of elements
            OperationMatrixAggregate AbsMax(void) const;
            OperationMatrixAggregate AbsMin(void) const;
            OperationMatrixAggregate AbsSum(void) const;

            //OperationMatrixTransform operator^(const char *inType) const;

            //operation with scalar
            OperationMatrixApplyElementwise operator^(float inVal) const;
            OperationMatrixApplyElementwise operator+(float inVal) const;
            OperationMatrixApplyElementwise operator-(float inVal) const;
            OperationMatrixApplyElementwise operator*(float inVal) const;
            OperationMatrixApplyElementwise operator/(float inVal) const;

            OperationMatrixApplyElementwise Sigmoid(void) const;
            OperationMatrixApplyElementwise Minimally(float inMin) const;
            OperationMatrixApplyElementwise Maximally(float inMax) const;

            MatrixGpu &operator^=(float inExponent);
            MatrixGpu &operator*=(float inVal);

        protected:
            void Init(t_index inX, t_index inY, bool inTransposed)
            {
                m_X = inX;
                m_Y = inY;
                m_Data.Reset(inX*inY);
                m_Transposed = inTransposed;
                m_ShallowCopy = false;
            }

            t_index m_X;
            t_index m_Y;
            GpuData<float> m_Data;
            bool m_Transposed;
            bool m_ShallowCopy;

#ifdef DEBUG_MATRIX_CLASS
            static int m_Allocations;
#endif

            friend OperationMatrixMultiply Mult(const MatrixGpu &inA, const MatrixGpu &inB);//matrix multiplication!
    };

#ifdef DEBUG_MATRIX_CLASS
int MatrixGpu::m_Allocations = 0;
#endif

    struct OptimizationInfo
    {
        //TODO: can this handle be used for optimization of operations?
        //It should aggregate all operations in the one command/assigment
        //Then rearrange/megre operations into more complex kernel if possigle
    };


    class OperationGpu
    {
        public:

            OperationGpu(void)
            {
                DEB_CONSTRUCTOR(OperationGpu);
            }
            virtual void GetOptimizationInfo(OptimizationInfo &inOutOptInfo)
            {
                //TODO: adds tree-based operaton sequence
            }

            virtual void Execute(MatrixGpu &outMatrix) const = 0;

            virtual void GetResultSize(t_index &outX, t_index &outY, bool &outTransposed) const = 0;

            virtual ~OperationGpu(void)
            {
                DEB_DESTRUCTOR(OperationGpu);
            }

        protected:
            static const float m_Zero;
            static const float m_One;
            static const float m_MinusOne;

    };

    const float OperationGpu::m_Zero = 0.0f;
    const float OperationGpu::m_One = 1.0f;
    const float OperationGpu::m_MinusOne = -1.0f;

    class OperationMatrixMultiply : public OperationGpu
    {
        public:
            OperationMatrixMultiply(const MatrixGpu& inA, const MatrixGpu& inB)
                : m_A(inA), m_B(inB)
            {
                t_index kA = !inA.isTrans() ? m_A.getY() : m_A.getX();
                t_index kB = !inB.isTrans() ? m_B.getX() : m_B.getY();
                //std::cout << "Mult:" << kA << " vs " << kB << std::endl;
                //std::cout << " " << m_A.getX() << "x" << m_A.getY() << (inA.isTrans() ? "T" : "") << std::endl;
                //std::cout << " " << m_B.getX() << "x" << m_B.getY() << (inB.isTrans() ? "T" : "") << std::endl;
                assert(kA == kB);
            }

            virtual void GetResultSize(t_index &outX, t_index &outY, bool &outTransposed) const
            {
                outX = !m_A.isTrans() ? m_A.getX() : m_A.getY();
                outY = !m_B.isTrans() ? m_B.getY() : m_B.getX();
                outTransposed = false;
            }

            virtual void Execute(MatrixGpu &outMatrix) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                t_index x = !m_A.isTrans() ? m_A.getX() : m_A.getY();
                t_index y = !m_B.isTrans() ? m_B.getY() : m_B.getX();
                t_index kA = !m_A.isTrans() ? m_A.getY() : m_A.getX();
                t_index kB = !m_B.isTrans() ? m_B.getX() : m_B.getY();

                //cout << "TA:" << m_A.isTrans() << ", TB:" << m_B.isTrans() << endl;
                assert(kA == kB);

                outMatrix.Reset(x, y);

                cublasHandle_t handle;
                cublasStatus_t stat = cublasCreate(&handle);

                assert (stat == CUBLAS_STATUS_SUCCESS);

                cublasSgemm(handle, !m_A.isTrans() ? CUBLAS_OP_N : CUBLAS_OP_T, !m_B.isTrans() ? CUBLAS_OP_N : CUBLAS_OP_T,
                        x, y, kA,
                        &m_One, m_A.getDataConst(), m_A.getX(), m_B.getDataConst(), m_B.getX(), &m_Zero, outMatrix.getDataConst(), x);

                cublasDestroy(handle);

            }

        protected:
            const MatrixGpu& m_A;
            const MatrixGpu& m_B;
    };

    __global__ void parallelAssociativeOperator(const float *inData, t_index inN, EFunctionBinaryAssociative inType, float *outBlockResults)
    {
        extern __shared__ float sd[];

        t_index idx = blockIdx.x*blockDim.x + threadIdx.x;

        float myData;
        switch(inType)
        {
        case EFB_Plus:
            myData = 0.0f;;
            break;
        case EFB_Multiply:
            myData = 1.0f;
            break;
        case EFB_Max:
            //myData = -std::numeric_limits<float>::max();
            myData = -FLT_MAX;
            break;
        case EFB_Min:
            //myData = std::numeric_limits<float>::max();
            myData = FLT_MAX;
            break;
        }

        //load to shared memory
        if(idx < inN)
        {
            myData = inData[idx];
        }

        sd[threadIdx.x] = myData;

        for(t_index offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            __syncthreads();

            if(threadIdx.x < offset)
            {
                switch(inType)
                {
//                case sd[threadIdx.x] += sd[threadIdx.x + offset];
                case EFB_Plus:
                    sd[threadIdx.x] += sd[threadIdx.x + offset];
                    break;
                case EFB_Multiply:
                    sd[threadIdx.x] *= sd[threadIdx.x + offset];
                    break;
                case EFB_Max:
                    if(sd[threadIdx.x] < sd[threadIdx.x + offset])
                    {
                        sd[threadIdx.x] = sd[threadIdx.x + offset];
                    }
                    break;
                case EFB_Min:
                    if(sd[threadIdx.x] > sd[threadIdx.x + offset])
                    {
                        sd[threadIdx.x] = sd[threadIdx.x + offset];
                    }
                    break;
                }

            }
        }

        //thread no. 0 stores result
        if(threadIdx.x == 0)
        {
            outBlockResults[blockIdx.x] = sd[0];
        }
    }
    class OperationBinaryAssociative : public OperationGpu
    {
        public:
            OperationBinaryAssociative(const MatrixGpu& inA, EFunctionBinaryAssociative inType)
                : m_A(inA), m_Type(inType)
            {
            }

            virtual void GetResultSize(t_index &outX, t_index &outY, bool &outTransposed) const
            {
                outX = 1;
                outY = 1;
                outTransposed = false;
            }

            virtual void Execute(MatrixGpu &outMatrix) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                outMatrix.Reset(1, 1);

                static t_index ThreadsPerBlock = 512;
                t_index num = m_A.getX()*m_A.getY();

                t_index blocks = (num - 1) / ThreadsPerBlock + 1;
                GpuData<float> tmp(blocks);
                parallelAssociativeOperator<<<blocks, ThreadsPerBlock, ThreadsPerBlock*sizeof(float)>>>(m_A.getDataConst(), num, m_Type, tmp.raw());

                while(blocks > ThreadsPerBlock)//repetitive call
                {
                    t_index num = blocks;
                    blocks = (num -1) / ThreadsPerBlock + 1;

                    GpuData<float> tmp2(blocks);

                    swapData(tmp, tmp2);

                    parallelAssociativeOperator<<<blocks, ThreadsPerBlock, ThreadsPerBlock*sizeof(float)>>>(tmp2.raw(), num, m_Type, tmp.raw());
                }

                parallelAssociativeOperator<<<1, blocks, blocks*sizeof(float)>>>(tmp.raw(), blocks, m_Type, outMatrix.getData());
            }

        protected:
            const MatrixGpu& m_A;
            const EFunctionBinaryAssociative m_Type;
    };


    __global__ void parallelMatrixOperationBinary(const float *inA, const float *inB, t_index inN, EFunctionElementwiseBinary inType, float *outResult)
    {
        //extern __shared__ float sd[];

        t_index idx = blockIdx.x*blockDim.x + threadIdx.x;

        //load to shared memory
        if(idx < inN)
        {
            switch(inType)
            {
            case EFEB_Plus:
                outResult[idx] = inA[idx] + inB[idx];
                break;
            case EFEB_Minus:
                outResult[idx] = inA[idx] - inB[idx];
                break;
            case EFEB_Multiply:
                outResult[idx] = inA[idx] * inB[idx];
                break;
            case EFEB_Divide:
                outResult[idx] = inA[idx] / inB[idx];
                break;
            }
        }
    }

    class OperationMatrixElementwiseBinary : public OperationGpu
    {
        public:
            OperationMatrixElementwiseBinary(const MatrixGpu& inA, const MatrixGpu& inB, EFunctionElementwiseBinary inType)
                : m_A(inA), m_B(inB), m_Type(inType)
            {
                //if(inA.getX() != inB.getX() || inA.getY() != inB.getY())
                //{
                //    cout << "wanted: " << inA.getX() << "x" << inA.getY() << ", got " << inB.getX() << "x" << inB.getY() << endl;
                //}
                assert (inA.getX() == inB.getX() && inA.getY() == inB.getY());
                assert (inA.isTrans() == inB.isTrans());
            }

            virtual void GetResultSize(t_index &outX, t_index &outY, bool &outTransposed) const
            {
                outX = m_A.getX();
                outY = m_A.getY();
                outTransposed = m_A.isTrans();
            }

            virtual void Execute(MatrixGpu &outMatrix) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                static t_index ThreadsPerBlock = 512;

                t_index num = m_A.getX()*m_A.getY();

                t_index blocks = (num - 1) / ThreadsPerBlock + 1;

                parallelMatrixOperationBinary<<<blocks, ThreadsPerBlock>>>(m_A.getDataConst(), m_B.getDataConst(), num, m_Type, outMatrix.getData());

                debugMatrix(outMatrix);
            }

        protected:
            const MatrixGpu& m_A;
            const MatrixGpu& m_B;
            const EFunctionElementwiseBinary m_Type;
    };


    __global__ void transposeSimple(float *outData, const float *inData, t_index inSourceX, t_index inSourceY)
    {
        extern __shared__ float tile[];

        t_index sharedIdx = threadIdx.x*blockDim.y + threadIdx.y;
        
        t_index sourceX = blockDim.x*blockIdx.x + threadIdx.x;
        t_index sourceY = blockDim.y*blockIdx.y + threadIdx.y;

        //range-check
        if(sourceX < inSourceX && sourceY < inSourceY)
        {
            t_index sourceIdx = sourceY*inSourceX + sourceX;
            t_index targetIdx = sourceX*inSourceY + sourceY;

            tile[sharedIdx] = inData[sourceIdx];
            
            __syncthreads();

            outData[targetIdx] = tile[sharedIdx];
        }
    }

    void transpose(float *outData, const float *inData, t_index inSourceX, t_index inSourceY)
        {
            static t_index ThreadsPerBlock = 32; //shared memory need to be at least sizeof(float) * ThreadsPerBlock^2

            t_index sx = min(ThreadsPerBlock, inSourceX);
            t_index sy = min(ThreadsPerBlock, inSourceY);

            dim3 threadsPerBlock(sx, sy, 1);

            dim3 blocksPerGrid(
                      (inSourceX - 1) / sx + 1
                    , (inSourceY - 1) / sy + 1
                    , 1);

            //std::cout << std::endl;
            //std::cout << "x = " << inSourceX << ", y = " << inSourceY << std::endl;
            //std::cout << "tpb: " << threadsPerBlock.x << " x " << threadsPerBlock.y << " x " << threadsPerBlock.z << std::endl;
            //std::cout << "bpg: " << blocksPerGrid.x << " x " << blocksPerGrid.y << " x " << blocksPerGrid.z << std::endl;

            //std::cout << std::endl;

            transposeSimple<<<blocksPerGrid, threadsPerBlock, sx*sy*sizeof(float)>>>(outData, inData, inSourceX, inSourceY);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }

#define TILE_DIM 32
#define BLOCK_ROWS 8

    __global__ void transposeCoalesced(float *odata, const float *idata, t_index inMaxX, t_index inMaxY)
    {
        __shared__ float tile[TILE_DIM][TILE_DIM+1];
        
        t_index x = blockIdx.x * TILE_DIM + threadIdx.x;
        t_index y = blockIdx.y * TILE_DIM + threadIdx.y;

        t_index width = gridDim.x * TILE_DIM;
        
        for (t_index j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
        }
        
        __syncthreads();
        
        x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
        y = blockIdx.x * TILE_DIM + threadIdx.y;
        
        for (t_index j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if((y+j) < inMaxY && width < inMaxX)
            {
                odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }

    __global__ void getIndexValue(const float *inData, t_index inIndex, float *outData)
    {
        *outData = inData[inIndex];
    }


    void testSum(const MatrixGpu& inA, MatrixGpu &outMatrix)
    {
    }

    class OperationMatrixAggregate : public OperationGpu
    {
        public:
            OperationMatrixAggregate(const MatrixGpu& inA, EAggregate inType)
                : m_A(inA), m_Type(inType)
            {
            }

            virtual void GetResultSize(t_index &outX, t_index &outY, bool &outTransposed) const
            {
                outX = 1;
                outY = 1;
                outTransposed = false;
            }

            virtual void Execute(MatrixGpu &outMatrix) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                outMatrix.Reset(1, 1);
                
                cublasHandle_t handle;
                cublasStatus_t stat = cublasCreate(&handle);

                assert (stat == CUBLAS_STATUS_SUCCESS);

                if(m_Type == EA_AbsSum)
                {
                    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
                    cublasSasum(handle, m_A.getX()*m_A.getY(), m_A.getDataConst(), 1, outMatrix.getData());
                }
                else if(m_Type == EA_AbsMin)
                {
                    //t_index resIndex;

                    //cublasIsamin(handle, m_A.getX()*m_A.getY(), m_A.getDataConst(), 1, &resIndex);

                    //getIndexValue<<<1, 1>>>(m_A.getDataConst(), resIndex-1, outMatrix.getData());//-1 because of fortran-style min function
                    //cudaDeviceSynchronize();
                    //cudaThreadSynchronize();

                    //not implemented - cublas int* vs i_index* conflict
                    assert(0);

                    //cout << "MIN_INDEX = " << resIndex << endl;
                }
                else if(m_Type == EA_AbsMax)
                {
                    //t_index resIndex;

                    //cublasIsamax(handle, m_A.getX()*m_A.getY(), m_A.getDataConst(), 1, &resIndex);

                    //getIndexValue<<<1, 1>>>(m_A.getDataConst(), resIndex-1, outMatrix.getData());

                    //cudaDeviceSynchronize();
                    //cudaThreadSynchronize();

                    //not implemented - cublas int* vs i_index* conflict
                    assert(0);

                    //cout << "MAX_INDEX = " << resIndex << endl;
                }
                else if(m_Type == EA_Sum)
                {
                    testSum(m_A, outMatrix);                    
                }

                cublasDestroy(handle);
            }

        protected:
            const MatrixGpu& m_A;
            const EAggregate m_Type;
    };

    __global__ void applyFunction(float *outTarget, const float *inSource, t_index N, EFunctionElementwise inType, float inParam1)
        {
            // target index
            t_index tid = blockDim.x * blockIdx.x + threadIdx.x;
        
            if (tid < N)
            {
                switch(inType)
                {
                    case EFE_Square:
                        outTarget[tid] = (inSource[tid]*inSource[tid]);
                        break;
                    case EFE_Sqrt:
                        outTarget[tid] = sqrtf(inSource[tid]);
                        break;
                    case EFE_Pow:
                        outTarget[tid] = powf(inSource[tid], inParam1);
                        break;
                    case EFE_ScalarMultiply:
                        outTarget[tid] = inParam1*inSource[tid];
                        break;
                    case EFE_Fill:
                        outTarget[tid] = inParam1;
                        break;
                    case EFE_Less:
                        outTarget[tid] = inSource[tid] < inParam1;
                        break;
                    case EFE_LessOrEqual:
                        outTarget[tid] = inSource[tid] <= inParam1;
                        break;
                    case EFE_Equal:
                        outTarget[tid] = inSource[tid] == inParam1;
                        break;
                    case EFE_GreaterOrEqual:
                        outTarget[tid] = inSource[tid] >= inParam1;
                        break;
                    case EFE_Greater:
                        outTarget[tid] = inSource[tid] > inParam1;
                        break;
                    case EFE_NotEqual:
                        outTarget[tid] = inSource[tid] != inParam1;
                        break;
                    case EFE_PlusScalar:
                        outTarget[tid] = inSource[tid] + inParam1;
                        break;
                    case EFE_MinusScalar:
                        outTarget[tid] = inSource[tid] - inParam1;
                        break;
                    case EFE_MultiplyScalar:
                        outTarget[tid] = inSource[tid] * inParam1;
                        break;
                    case EFE_DivideScalar:
                        outTarget[tid] = inSource[tid] / inParam1;
                        break;
                    case EFE_InverseAndMultiply:
                        outTarget[tid] = inParam1 / inSource[tid];
                        break;
                    case EFE_Sigmoid:
                        outTarget[tid] = 1.0f/(1.0f + expf(-inSource[tid]));
                        break;
                    case EFE_Minimally:
                        outTarget[tid] = max(0.0f, inSource[tid]);
                        break;
                    case EFE_Maximally:
                        outTarget[tid] = min(0.0f, inSource[tid]);
                        break;
                }
            }
        }
    
    void funcElementwise(MatrixGpu &outMatrix, const MatrixGpu &inMatrix, EFunctionElementwise inType, float inParam1 = 0.0f)
        {
            assert (inMatrix.getX() == outMatrix.getX());
            assert (inMatrix.getY() == outMatrix.getY());

            static t_index ThreadsPerBlock = 512;

            dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);

            t_index num = inMatrix.getX()*inMatrix.getY();

            dim3 blocksPerGrid((num - 1) / ThreadsPerBlock + 1, 1, 1);

            applyFunction<<<blocksPerGrid, threadsPerBlock>>>(outMatrix.getData(), inMatrix.getDataConst(), num, inType, inParam1);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }

    class OperationMatrixApplyElementwise : public OperationGpu
    {
        public:
            OperationMatrixApplyElementwise(const MatrixGpu& inA, EFunctionElementwise inType, float inParam1)
                : m_A(inA), m_Type(inType), m_Param1(inParam1)
            {
            }

            virtual void GetResultSize(t_index &outX, t_index &outY, bool &outTransposed) const
            {
                outX = m_A.getX();
                outY = m_A.getY();
                outTransposed = m_A.isTrans();
            }

            virtual void Execute(MatrixGpu &outMatrix) const
            {
                funcElementwise(outMatrix, m_A, m_Type, m_Param1);
            }

        protected:
            const MatrixGpu& m_A;
            EFunctionElementwise m_Type;
            float m_Param1;
    };

    MatrixCpu::MatrixCpu(const MatrixGpu &inMatrix)
        : m_X(0), m_Y(0), m_Data(NULL), m_CacheFileName(""), m_FileCache(-1)
        {
            Reset(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToHost);
        }

    MatrixGpu::MatrixGpu(const MatrixGpu &inMatrix, bool inShallowCopy)
        {
#ifdef DEBUG_MATRIX_CLASS
            m_Allocations += 1;
            cout << "m" << m_Allocations;
#endif
            Init(inMatrix.getX(), inMatrix.getY(), inMatrix.isTrans());
            if(inShallowCopy)
            {
                m_Data = inMatrix.m_Data;
                m_ShallowCopy = true;
            }
            else
            {
                cudaMemcpy(getData(), inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }

    MatrixGpu::MatrixGpu(const MatrixCpu &inMatrix)
        {
#ifdef DEBUG_MATRIX_CLASS
            m_Allocations += 1;
            cout << "m" << m_Allocations;
#endif
            Init(inMatrix.getX(), inMatrix.getY(), false);
            cudaMemcpy(getData(), inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyHostToDevice);
        }

    MatrixGpu& MatrixGpu::operator=(const MatrixCpu &inMatrix)
        {
            Reset(inMatrix.getX(), inMatrix.getY(), false);
            cudaMemcpy(getData(), inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyHostToDevice);

            return *this;
        }
    MatrixGpu& MatrixGpu::operator=(const MatrixGpu &inMatrix)
        {
            if(this != &inMatrix)
            {
                Reset(inMatrix.getX(), inMatrix.getY(), inMatrix.isTrans());
                cudaMemcpy(getData(), inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToDevice);
            }

            return *this;
        }
    MatrixGpu& MatrixGpu::operator=(const OperationGpu &inOperation)
        {
            t_index x, y;
            bool trans;
            inOperation.GetResultSize(x, y, trans);
            Init(x, y, trans);


            //TODO:optimization step
            //OptimizationInfo optInfo;
            //inOperation.GetOptimizationInfo(optInfo);
            //OperationGpu newOperation = Optimize(optInfo);
            //newOperation.Execute(*this);
 
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            inOperation.Execute(*this);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            return *this;
        }
    MatrixGpu &MatrixGpu::operator=(float inFill)
        {
            funcElementwise(*this, *this, EFE_Fill, inFill);

            return *this;
        }

    MatrixGpu::MatrixGpu(const OperationGpu &inOperation)
        {
#ifdef DEBUG_MATRIX_CLASS
            m_Allocations += 1;
            cout << "m" << m_Allocations;
#endif
            t_index x, y;
            bool trans;
            inOperation.GetResultSize(x, y, trans);
            Init(x, y, trans);

            //TODO:optimization step
            //OptimizationInfo optInfo;
            //inOperation.GetOptimizationInfo(optInfo);
            //OperationGpu newOperation = Optimize(optInfo);
            //newOperation.Execute(*this);
 
            inOperation.Execute(*this);
        }

    OperationMatrixMultiply Mult(const MatrixGpu &inA, const MatrixGpu &inB)//matrix multiplication friend!
        {
            return OperationMatrixMultiply(inA, inB);
        }

    OperationMatrixAggregate MatrixGpu::AbsMax(void) const
        {
            return OperationMatrixAggregate(*this, EA_AbsMax);
        }
    OperationMatrixAggregate MatrixGpu::AbsMin(void) const
        {
            return OperationMatrixAggregate(*this, EA_AbsMin);
        }
    OperationMatrixAggregate MatrixGpu::AbsSum(void) const
        {
            return OperationMatrixAggregate(*this, EA_AbsSum);
        }
    OperationBinaryAssociative MatrixGpu::Sum(void) const
        {
            return OperationBinaryAssociative(*this, EFB_Plus);
        }

    OperationBinaryAssociative MatrixGpu::Product(void) const
        {
            return OperationBinaryAssociative(*this, EFB_Multiply);
        }

    OperationBinaryAssociative MatrixGpu::Max(void) const
        {
            return OperationBinaryAssociative(*this, EFB_Max);
        }

    OperationBinaryAssociative MatrixGpu::Min(void) const
        {
            return OperationBinaryAssociative(*this, EFB_Min);
        }

    MatrixGpu &MatrixGpu::operator^=(float inExponent)
    {
        if(inExponent == 2.0f)
        {
            return this->operator=(OperationMatrixApplyElementwise(*this, EFE_Square, 0.0f));
        }
        else if(inExponent == 0.5f)
        {
            return this->operator=(OperationMatrixApplyElementwise(*this, EFE_Sqrt, 0.0f));
        }
        else
        {
            return this->operator=(OperationMatrixApplyElementwise(*this, EFE_Pow, inExponent));
        }
    }

    MatrixGpu &MatrixGpu::operator*=(float inVal)
    {
        return this->operator=(OperationMatrixApplyElementwise(*this, EFE_ScalarMultiply, inVal));
    }

    OperationMatrixElementwiseBinary MatrixGpu::operator+(const MatrixGpu &inB) const
    {
        return OperationMatrixElementwiseBinary(*this, inB, EFEB_Plus);
    }

    OperationMatrixElementwiseBinary MatrixGpu::operator-(const MatrixGpu &inB) const
    {
        return OperationMatrixElementwiseBinary(*this, inB, EFEB_Minus);
    }

    OperationMatrixElementwiseBinary MatrixGpu::operator*(const MatrixGpu &inB) const
    {
        return OperationMatrixElementwiseBinary(*this, inB, EFEB_Multiply);
    }

    OperationMatrixElementwiseBinary MatrixGpu::operator/(const MatrixGpu &inB) const
    {
        return OperationMatrixElementwiseBinary(*this, inB, EFEB_Divide);
    }

    MatrixGpu &MatrixGpu::operator+=(const MatrixGpu &inB) 
    {
        return this->operator=(OperationMatrixElementwiseBinary(*this, inB, EFEB_Plus));
    }

    MatrixGpu &MatrixGpu::operator-=(const MatrixGpu &inB)
    {
        return this->operator=(OperationMatrixElementwiseBinary(*this, inB, EFEB_Minus));
    }

    MatrixGpu &MatrixGpu::operator*=(const MatrixGpu &inB)
    {
        return this->operator=(OperationMatrixElementwiseBinary(*this, inB, EFEB_Multiply));
    }

    MatrixGpu &MatrixGpu::operator/=(const MatrixGpu &inB)
    {
        return this->operator=(OperationMatrixElementwiseBinary(*this, inB, EFEB_Divide));
    }


    OperationMatrixApplyElementwise MatrixGpu::operator<(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_Less, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::operator<=(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_LessOrEqual, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::operator==(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_Equal, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::operator>=(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_GreaterOrEqual, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::operator>(float  inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_Greater, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::operator!=(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_NotEqual, inVal);
    }

    OperationMatrixApplyElementwise MatrixGpu::operator^(float inVal) const
    {
        if(inVal == 2.0f)
        {
            return OperationMatrixApplyElementwise(*this, EFE_Square, 0.0f);
        }
        else if(inVal == 0.5f)
        {
            return OperationMatrixApplyElementwise(*this, EFE_Square, 0.0f);
        }
        else
        {
            return OperationMatrixApplyElementwise(*this, EFE_Pow, inVal);
        }
    }
    OperationMatrixApplyElementwise MatrixGpu::operator+(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_PlusScalar, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::operator-(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_MinusScalar, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::operator*(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_MultiplyScalar, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::operator/(float inVal) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_DivideScalar, inVal);
    }
    OperationMatrixApplyElementwise MatrixGpu::Sigmoid(void) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_Sigmoid, 0.0f);
    }

    OperationMatrixApplyElementwise MatrixGpu::Minimally(float inMin) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_Minimally, inMin);
    }

    OperationMatrixApplyElementwise MatrixGpu::Maximally(float inMax) const
    {
        return OperationMatrixApplyElementwise(*this, EFE_Maximally, inMax);
    }

    __global__ void sample(float *outData, float *inData, float* inRnd, t_index x, t_index y, t_index N)
    {
        //target row id
        t_index rid = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (rid < N)
        {
            //random row-id
            t_index row = int(inRnd[rid]*x);

            for(t_index i = 0; i < y; ++i)
            {
                outData[IDX2C(rid, i, N)] = inData[IDX2C(row, i, x)];
            }

        }
    }

    void MatrixGpu::Sample(t_index inRowsNum, MatrixGpu &outSample) const
    {
        MatrixGpu rnd(inRowsNum, 1);
        rnd.RandUniform();

        outSample.Reset(inRowsNum, getY());

        static t_index ThreadsPerBlock = 256;

        dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);

        dim3 blocksPerGrid((inRowsNum - 1) / ThreadsPerBlock + 1, 1, 1);

        sample<<<blocksPerGrid, threadsPerBlock>>>(outSample.getData(), getDataConst(), rnd.getDataConst(), getX(), getY(), inRowsNum);
    }

    void MatrixGpu::MakeHardCopy(void)
    {
        if(m_ShallowCopy || isTrans())
        {
            GpuData<float> sourceData = m_Data;
            m_Data.Reset();

            if(isTrans())
            {
                t_index x = getX();
                t_index y = getY();

                //transpose
                Init(y, x, false);

                transpose(getData(), sourceData.raw(), x, y);
            }
            else
            {
                //copy-only
                Init(getX(), getY(), false);

                cudaMemcpy(getData(), sourceData.raw(), getX()*getY()*sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }
    }

    MatrixCpu &MatrixCpu::operator=(const MatrixGpu &inMatrix)
    {
        Reset(inMatrix.getX(), inMatrix.getY());
        assert(!inMatrix.isTrans());
        cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToHost);

        return *this;
    }

}

#endif //MATRIX_H

