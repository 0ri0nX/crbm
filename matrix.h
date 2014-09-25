#ifndef MATRIX_H
#define MATRIX_H

#include <cublas_v2.h>
#include <curand.h>
#include <limits>
#include <float.h>

namespace YAMATH
{

//#define DEBUG_MATRIX_CLASS
//#define DEBUG_ALLOCATION

//#define debugMatrix(m) { std::cout << "matrix at " << __LINE__ << ": " << (m).getX() << " x " << (m).getY() << ((m).isTrans() ? "T" : "") <<  std::endl;}
#define debugMatrix(m) 


#ifdef DEBUG_ALLOCATION
    int xxx = 0;
#endif

    template <typename T>
    inline T* allocateGpu(int inNum)
    {
#ifdef DEBUG_ALLOCATION
        ++xxx;
        cout << "a(" << xxx << ")";
#endif
        T *ptr = NULL;
        cudaMalloc((void**) &ptr, inNum*sizeof(T));
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
            Holder(int inNum) :
                m_Counter(1), m_Data(NULL)
            {
                cudaMalloc((void**) &m_Data, inNum*sizeof(T));
                assert(m_Data != NULL);
            }

            ~Holder(void)
            {
                assert(m_Counter == 0);
                cudaFree(m_Data);
            }
    
            int m_Counter;
            float *m_Data;
        };

        public:
            GpuData(int inNum = 0)
            {
                assert(inNum >= 0);

                if(inNum != 0)
                {
                    m_Holder = new Holder(inNum);
                }
                else
                {
                    m_Holder = NULL;
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

            GpuData<T> &operator=(int inNum)
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

            void Reset(int inNum = 0)
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
    };
/*
#ifdef DEBUG_ALLOCATION
    int xxx = 0;
#endif

    float * allocate(int inNum)
    {
#ifdef DEBUG_ALLOCATION
        ++xxx;
        cout << "a(" << xxx << ")";
#endif
        float *ptr = NULL;
        cudaMalloc((void**) &ptr, inNum*sizeof(float));
        assert(ptr != NULL);
        return ptr;
    }

    int * allocateInt(int inNum)
    {
#ifdef DEBUG_ALLOCATION
        cout << "ai";
#endif
        int *ptr = NULL;
        cudaMalloc((void**) &ptr, inNum*sizeof(int));
        assert(ptr != NULL);
        return ptr;
    }

    void deallocate(float *inFloatArray)
    {
#ifdef DEBUG_ALLOCATION
        --xxx;
        cout << "d";
#endif
        cudaFree(inFloatArray);
    }

    void deallocateInt(int *inIntArray)
    {
        cout << "di";
        cudaFree(inIntArray);
    }
*/

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
    //class OperationMatrixTransform;
    class OperationBinaryAssociative;
    class OperationMatrixElementwiseBinary;

    class MatrixCpu//column-first layout
    {
        public:
            MatrixCpu(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
                : m_X(inX), m_Y(inY), m_Data(0)
            {
                Init(inX, inY, inInit);
            }

            MatrixCpu(const MatrixGpu &inMatrix);
            MatrixCpu(const MatrixCpu &inMatrix);

            bool Load(istream &inStream, bool inTransposed = false)
            {
//column-first order - ld is leading dimension size - #rows
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

                int x, y;
                inStream >> x >> y;

                assert (x >= 0 && y >= 0);

                //cout << "x:" << x << "\ny:" << y << std::endl;

                if(!inTransposed)
                {
                    Reset(x, y);
                    for(int i = 0; i < x; ++i)
                    {
                        for(int j = 0; j < y; ++j)
                        {
                            inStream >> m_Data[IDX2C(i, j, x)];
                        }
                    }
                }
                else
                {
                    Reset(y, x);
                    for(int i = 0; i < x; ++i)
                    {
                        for(int j = 0; j < y; ++j)
                        {
                            inStream >> m_Data[IDX2C(j, i, y)];
                        }
                    }
                }


                return true;
            }

            void Save(ostream &outStream) const
            {
                outStream << m_X << " " << m_Y << std::endl;
                for(int i = 0; i < m_X; ++i)
                {
                    if(m_Y > 0)
                    {
                        outStream << m_Data[IDX2C(i, 0, m_X)];
                    }

                    for(int j = 1; j < m_Y; ++j)
                    {
                        outStream << " " << m_Data[IDX2C(i, j, m_X)];
                    }
                    outStream << std::endl;
                }
            }

            ~MatrixCpu(void)
            {
                delete [] m_Data;
            }

            int getX(void) const { return m_X; }
            int getY(void) const { return m_Y; }
            float* getDataConst(void) const { return m_Data; }
            float* getData(void) { return m_Data; }

            inline void set(int inX, int inY, float inValue)
            {
                m_Data[IDX2C(inX, inY, getX())] = inValue;
            }

            inline float get(int inX, int inY) const
            {
                return m_Data[IDX2C(inX, inY, getX())];
            }

            

            MatrixCpu SubMatrix(int inStartRow, int inStartCol, int inEndRow, int inEndCol) //start is inclusive, end is NON inclusive
            {
                assert(inStartRow >= 0 && inEndRow <= getX());
                assert(inStartCol >= 0 && inEndCol <= getY());

                MatrixCpu m(inEndRow - inStartRow, inEndCol - inStartCol);

                for(int i = 0; i < m.getX(); ++i)
                {
                    for(int j = 0; j < m.getY(); ++j)
                    {
                        m.set(i, j, get(inStartRow + i, inStartCol + j));
                    }
                }

                return m;
            }

            void Reset(int inX, int inY, const float * inInit = NULL)
            {
                if(m_X != inX || m_Y != inY)
                {
                    delete [] m_Data;
                    Init(inX, inY, inInit);
                }
            }

        protected:
            void Init(int inX, int inY, const float *inInit = NULL)
            {
                assert (inX > 0 && inY > 0);

                m_Data = new float [inX*inY];
                m_X = inX;
                m_Y = inY;

                if(inInit != NULL)
                {
                    memcpy(m_Data, inInit, inX*inY*sizeof(float));
                }
            }

            int m_X;
            int m_Y;
            float *m_Data;
    };



    class MatrixGpu
    {
        public:
            MatrixGpu(void)
                : m_X(1), m_Y(1), m_Data(0), m_Transposed(false)
            {
                m_Data.Reset(1);
#ifdef DEBUG_MATRIX_CLASS
                cout << "c";
#endif
            }


            MatrixGpu(int x, int y, bool inTransposed = false)
                : m_X(x), m_Y(y), m_Data(0), m_Transposed(inTransposed)
            {
                m_Data.Reset(m_X*m_Y);
#ifdef DEBUG_MATRIX_CLASS
                cout << "c";
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
                cout << "d";
#endif
                //deallocate(m_Data);
            }

            int getX(void) const { return m_X; }
            int getY(void) const { return m_Y; }
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

            void Reset(int inX, int inY, bool inTransposed = false)
            {
                if(m_X != inX || m_Y != inY)
                {
                    m_Data = 0;

                    Init(inX, inY, inTransposed);
                }
                else
                {
                    m_Transposed = inTransposed;
                }
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

            MatrixGpu &operator^=(float inExponent);
            MatrixGpu &operator*=(float inVal);

        protected:
            void Init(int inX, int inY, bool inTransposed)
            {
                m_X = inX;
                m_Y = inY;
                m_Data.Reset(inX*inY);
                m_Transposed = inTransposed;
            }

            int m_X;
            int m_Y;
            GpuData<float> m_Data;
            bool m_Transposed;

            friend OperationMatrixMultiply Mult(const MatrixGpu &inA, const MatrixGpu &inB);//matrix multiplication!
    };



    class OperationGpu
    {
        public:
            OperationGpu(void){}

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const = 0;

            virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const = 0;

            virtual ~OperationGpu(void){}

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
                int kA = !inA.isTrans() ? m_A.getY() : m_A.getX();
                int kB = !inB.isTrans() ? m_B.getX() : m_B.getY();
                //std::cout << "Mult:" << kA << " vs " << kB << std::endl;
                //std::cout << " " << m_A.getX() << "x" << m_A.getY() << (inA.isTrans() ? "T" : "") << std::endl;
                //std::cout << " " << m_B.getX() << "x" << m_B.getY() << (inB.isTrans() ? "T" : "") << std::endl;
                assert(kA == kB);
            }

            virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const
            {
                outX = !m_A.isTrans() ? m_A.getX() : m_A.getY();
                outY = !m_B.isTrans() ? m_B.getY() : m_B.getX();
                outTransposed = false;
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                int x = !m_A.isTrans() ? m_A.getX() : m_A.getY();
                int y = !m_B.isTrans() ? m_B.getY() : m_B.getX();
                int kA = !m_A.isTrans() ? m_A.getY() : m_A.getX();
                int kB = !m_B.isTrans() ? m_B.getX() : m_B.getY();

                //cout << "TA:" << m_A.isTrans() << ", TB:" << m_B.isTrans() << endl;
                assert(kA == kB);

                outMatrix.Reset(x, y);

                cublasSgemm(inHandle, !m_A.isTrans() ? CUBLAS_OP_N : CUBLAS_OP_T, !m_B.isTrans() ? CUBLAS_OP_N : CUBLAS_OP_T,
                        x, y, kA,
                        &m_One, m_A.getDataConst(), m_A.getX(), m_B.getDataConst(), m_B.getX(), &m_Zero, outMatrix.getDataConst(), x);

                //cublasSgemm(inHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                //        m_A.getX(), m_B.getY(), m_A.getY(),
                //        &m_One, m_A.getDataConst(), m_A.getX(), m_B.getDataConst(), m_B.getX(), &m_Zero, outMatrix.getDataConst(), m_A.getX());
            }

        protected:
            const MatrixGpu& m_A;
            const MatrixGpu& m_B;
    };

    __global__ void parallelAssociativeOperator(const float *inData, int inN, EFunctionBinaryAssociative inType, float *outBlockResults)
    {
        extern __shared__ float sd[];

        unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

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

        for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
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

            virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const
            {
                outX = 1;
                outY = 1;
                outTransposed = false;
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                outMatrix.Reset(1, 1);


                static int ThreadsPerBlock = 512;

                int num = m_A.getX()*m_A.getY();

                int blocks = (num - 1) / ThreadsPerBlock + 1;

                //TODO: data should be held by object!!
                //float *tmp = allocateGpu<float>(blocks);
                GpuData<float> tmp(blocks);

                parallelAssociativeOperator<<<blocks, ThreadsPerBlock, ThreadsPerBlock*sizeof(float)>>>(m_A.getDataConst(), num, m_Type, tmp.raw());
                parallelAssociativeOperator<<<1, blocks, blocks*sizeof(float)>>>(tmp.raw(), blocks, m_Type, outMatrix.getData());
            }

        protected:
            const MatrixGpu& m_A;
            const EFunctionBinaryAssociative m_Type;
    };


    __global__ void parallelMatrixOperationBinary(const float *inA, const float *inB, int inN, EFunctionElementwiseBinary inType, float *outResult)
    {
        //extern __shared__ float sd[];

        unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

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
                assert (inA.getX() == inB.getX() && inA.getY() == inB.getY());
                assert (inA.isTrans() == inB.isTrans());
            }

            virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const
            {
                outX = m_A.getX();
                outY = m_A.getY();
                outTransposed = m_A.isTrans();
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                static int ThreadsPerBlock = 512;

                int num = m_A.getX()*m_A.getY();

                int blocks = (num - 1) / ThreadsPerBlock + 1;

                //parallelMatrixOperationBinary<<<blocks, ThreadsPerBlock/*, 2*ThreadsPerBlock*sizeof(float)*/>>>(m_A.getDataConst(), m_B.getDataCOnst(), num, m_Type, outMatrix.getData());
                parallelMatrixOperationBinary<<<blocks, ThreadsPerBlock>>>(m_A.getDataConst(), m_B.getDataConst(), num, m_Type, outMatrix.getData());

                debugMatrix(outMatrix);
            }

        protected:
            const MatrixGpu& m_A;
            const MatrixGpu& m_B;
            const EFunctionElementwiseBinary m_Type;
    };

    //class OperationMatrixAdd : public OperationGpu
    //{
    //    public:
    //        OperationMatrixAdd(const MatrixGpu& inA, const MatrixGpu& inB)
    //            : m_A(inA), m_B(inB)
    //        {
    //            assert (inA.getX() == inB.getX() && inA.getY() == inB.getY());
    //        }

    //        virtual void GetResultSize(int &outX, int &outY) const
    //        {
    //            outX = m_A.getX();
    //            outY = m_A.getY();
    //        }

    //        virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
    //        {
    //            //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

    //            outMatrix.Reset(m_A.getX(), m_A.getY());

    //            cublasSgeam(inHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    //                    m_A.getX(), m_A.getY(),
    //                    &m_One, m_A.getDataConst(), m_A.getX(),
    //                    &m_One, m_B.getDataConst(), m_B.getX(),
    //                    outMatrix.getDataConst(), m_A.getX());
    //        }

    //    protected:
    //        const MatrixGpu& m_A;
    //        const MatrixGpu& m_B;
    //};

#define TILE_DIM 32
#define BLOCK_ROWS 8

    __global__ void transposeCoalesced(float *odata, const float *idata, int inMaxX, int inMaxY)
    {
        __shared__ float tile[TILE_DIM][TILE_DIM+1];
        
        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int width = gridDim.x * TILE_DIM;
        
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
        }
        
        __syncthreads();
        
        x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
        y = blockIdx.x * TILE_DIM + threadIdx.y;
        
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if((y+j) < inMaxY && width < inMaxX)
            {
                odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }

    //class OperationMatrixTransform : public OperationGpu
    //{
    //    public:
    //        OperationMatrixTransform(const MatrixGpu& inA, const char *inType)
    //            : m_A(inA), m_Type(inType)
    //        {
    //            //only transpose now
    //            assert(inType[0] == 'T' && inType[1] == '\0');
    //        }

    //        virtual void GetResultSize(int &outX, int &outY) const
    //        {
    //            outX = m_A.getY();
    //            outY = m_A.getX();
    //        }

    //        virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
    //        {
    //            //static int ThreadsPerBlock = 32;
    //
    //            //dim3 threadsPerBlock(ThreadsPerBlock, ThreadsPerBlock, 1);
    //
    //            //dim3 threadsPerGrid((outMatrix.getX() - 1) / ThreadsPerBlock + 1, (outMatrix.getY() - 1) / ThreadsPerBlock + 1, 1);
    //
    //            //transposeCoalesced<<<threadsPerGrid, threadsPerBlock>>>(outMatrix.getData(), m_A.getDataConst(), outMatrix.getX(), outMatrix.getY());
    //        }

    //        OperationMatrixMultiply operator*(const MatrixGpu &inB) const;

    //    protected:
    //        const MatrixGpu& m_A;
    //        bool m_Type;
    //};

    __global__ void getIndexValue(const float *inData, int inIndex, float *outData)
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

            virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const
            {
                outX = 1;
                outY = 1;
                outTransposed = false;
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                outMatrix.Reset(1, 1);
                
                if(m_Type == EA_AbsSum)
                {
                    cublasSetPointerMode(inHandle, CUBLAS_POINTER_MODE_DEVICE);
                    cublasSasum(inHandle, m_A.getX()*m_A.getY(), m_A.getDataConst(), 1, outMatrix.getData());
                }
                else if(m_Type == EA_AbsMin)
                {
                    int resIndex;

                    cublasIsamin(inHandle, m_A.getX()*m_A.getY(), m_A.getDataConst(), 1, &resIndex);

                    //cudaDeviceSynchronize();
                    //cudaThreadSynchronize();

                    getIndexValue<<<1, 1>>>(m_A.getDataConst(), resIndex-1, outMatrix.getData());//-1 because of fortran-style min function

                    //cudaDeviceSynchronize();
                    //cudaThreadSynchronize();

                    //cout << "MIN_INDEX = " << resIndex << endl;
                }
                else if(m_Type == EA_AbsMax)
                {
                    int resIndex;

                    cublasIsamax(inHandle, m_A.getX()*m_A.getY(), m_A.getDataConst(), 1, &resIndex);

                    //cudaDeviceSynchronize();
                    //cudaThreadSynchronize();

                    getIndexValue<<<1, 1>>>(m_A.getDataConst(), resIndex-1, outMatrix.getData());

                    //cudaDeviceSynchronize();
                    //cudaThreadSynchronize();

                    //cout << "MAX_INDEX = " << resIndex << endl;
                }
                else if(m_Type == EA_Sum)
                {
                    testSum(m_A, outMatrix);                    
                }

            }

        protected:
            const MatrixGpu& m_A;
            const EAggregate m_Type;
    };

    __global__ void applyFunction(float *outTarget, const float *inSource, int N, EFunctionElementwise inType, float inParam1)
        {
            /* which element does this compute? */
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
        
            /* if valid, squre the array element */
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
                }
            }
        }
    
    void funcElementwise(MatrixGpu &outMatrix, const MatrixGpu &inMatrix, EFunctionElementwise inType, float inParam1 = 0.0f)
        {
            assert (inMatrix.getX() == outMatrix.getX());
            assert (inMatrix.getY() == outMatrix.getY());

            static int ThreadsPerBlock = 256;

            dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);

            int num = inMatrix.getX()*inMatrix.getY();

            dim3 threadsPerGrid((num - 1) / ThreadsPerBlock + 1, 1, 1);

            applyFunction<<<threadsPerGrid, threadsPerBlock>>>(outMatrix.getData(), inMatrix.getDataConst(), num, inType, inParam1);
            //gpuErrCheck( cudaPeekAtLastError() );
            //gpuErrCheck( cudaDeviceSynchronize() );
        }

    class OperationMatrixApplyElementwise : public OperationGpu
    {
        public:
            OperationMatrixApplyElementwise(const MatrixGpu& inA, EFunctionElementwise inType, float inParam1)
                : m_A(inA), m_Type(inType), m_Param1(inParam1)
            {
            }

            virtual void GetResultSize(int &outX, int &outY, bool &outTransposed) const
            {
                outX = m_A.getX();
                outY = m_A.getY();
                outTransposed = m_A.isTrans();
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                funcElementwise(outMatrix, m_A, m_Type, m_Param1);
            }

        protected:
            const MatrixGpu& m_A;
            EFunctionElementwise m_Type;
            float m_Param1;
    };

    //class OperationMatrixSubstract : public OperationGpu
    //{
    //    public:
    //        OperationMatrixSubstract(const MatrixGpu& inA, const MatrixGpu& inB)
    //            : m_A(inA), m_B(inB)
    //        {
    //            assert (inA.getX() == inB.getX() && inA.getY() == inB.getY());
    //        }

    //        virtual void GetResultSize(int &outX, int &outY) const
    //        {
    //            outX = m_A.getX();
    //            outY = m_A.getY();
    //        }

    //        virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
    //        {
    //            //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

    //            outMatrix.Reset(m_A.getX(), m_A.getY());

    //            cublasSgeam(inHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    //                    m_A.getX(), m_A.getY(),
    //                    &m_One, m_A.getDataConst(), m_A.getX(),
    //                    &m_MinusOne, m_B.getDataConst(), m_B.getX(),
    //                    outMatrix.getDataConst(), m_A.getX());
    //        }

    //    protected:
    //        const MatrixGpu& m_A;
    //        const MatrixGpu& m_B;
    //};

    MatrixCpu::MatrixCpu(const MatrixGpu &inMatrix)
        {
            Init(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToHost);
        }

    MatrixCpu::MatrixCpu(const MatrixCpu &inMatrix)
        {
            Init(inMatrix.getX(), inMatrix.getY(), inMatrix.getDataConst());
            //memcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float));
        }

    MatrixGpu::MatrixGpu(const MatrixGpu &inMatrix, bool inShallowCopy)
        {
            //cout << "c";
            Init(inMatrix.getX(), inMatrix.getY(), inMatrix.isTrans());
            if(inShallowCopy)
            {
                m_Data = inMatrix.m_Data;
            }
            else
            {
                cudaMemcpy(getData(), inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }

    MatrixGpu::MatrixGpu(const MatrixCpu &inMatrix)
        {
            //cout << "c";
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
            Reset(inMatrix.getX(), inMatrix.getY(), inMatrix.isTrans());
            cudaMemcpy(getData(), inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToDevice);

            return *this;
        }
    MatrixGpu& MatrixGpu::operator=(const OperationGpu &inOperation)
        {
            cublasHandle_t handle;
            int x, y;
            bool trans;
            inOperation.GetResultSize(x, y, trans);
            Init(x, y, trans);

            cublasStatus_t stat;
        
            stat = cublasCreate(&handle);
            assert (stat == CUBLAS_STATUS_SUCCESS);

            inOperation.Execute(*this, handle);

            cublasDestroy(handle);

            return *this;
        }
    MatrixGpu &MatrixGpu::operator=(float inFill)
        {
            funcElementwise(*this, *this, EFE_Fill, inFill);

            return *this;
        }

    MatrixGpu::MatrixGpu(const OperationGpu &inOperation)
        {
            int x, y;
            bool trans;
            inOperation.GetResultSize(x, y, trans);
            Init(x, y, trans);

            cublasHandle_t handle;
            cublasStatus_t stat;
        
            stat = cublasCreate(&handle);
            assert (stat == CUBLAS_STATUS_SUCCESS);

            inOperation.Execute(*this, handle);
        }

    OperationMatrixMultiply Mult(const MatrixGpu &inA, const MatrixGpu &inB)//matrix multiplication friend!
        {
            return OperationMatrixMultiply(inA, inB);
        }

    //OperationMatrixMultiply OperationMatrixTransform::operator*(const MatrixGpu &inB) const
    //    {
    //        return OperationMatrixMultiply(m_A, inB, m_Type, false);
    //    }

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
    //OperationMatrixTransform MatrixGpu::operator^(const char *inType) const
    //    {
    //        return OperationMatrixTransform(*this, inType);
    //    }
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
}

#endif //MATRIX_H
