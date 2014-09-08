#ifndef MATRIX_H
#define MATRIX_H

#include <cublas_v2.h>
#include <curand.h>
#include <limits>
#include <float.h>

namespace YAMATH
{
    int xxx = 50;

    float * allocate(int inNum)
    {
        assert(--xxx);
        cout << "af";
        float *ptr = NULL;
        cudaMalloc((void**) &ptr, inNum*sizeof(float));
        assert(ptr != NULL);
        return ptr;
    }

    int * allocateInt(int inNum)
    {
        cout << "ai";
        int *ptr = NULL;
        cudaMalloc((void**) &ptr, inNum*sizeof(int));
        assert(ptr != NULL);
        return ptr;
    }

    void deallocate(float *inFloatArray)
    {
        cout << "df";
        cudaFree(inFloatArray);
    }

    void deallocateInt(int *inIntArray)
    {
        cout << "di";
        cudaFree(inIntArray);
    }

    enum EFunctionElementwise
    {
        EFE_Square,
        EFE_Sqrt,
        EFE_Pow,
        EFE_ScalarMultiply,
        EFE_Fill,
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
    class OperationMatrixTransform;
    class OperationBinaryAssociative;

    class MatrixCpu//column-first layout
    {
        public:
            MatrixCpu(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
                : m_X(inX), m_Y(inY), m_Data(NULL)
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
            MatrixGpu(int x = 1, int y = 1)
                : m_X(x), m_Y(y), m_Data(NULL)
            {
                m_Data = allocate(m_X*m_Y);
                cout << "c";
            }

//column-first order - ld is leading dimension size - #rows
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

            MatrixGpu(const MatrixGpu &inMatrix);
            MatrixGpu(const MatrixCpu &inMatrix);
            MatrixGpu(const OperationGpu &inOperation);

            ~MatrixGpu(void)
            {
                cout << "d";
                deallocate(m_Data);
            }

            int getX(void) const { return m_X; }
            int getY(void) const { return m_Y; }
            float* getDataConst(void) const { return m_Data; }
            float* getData(void) { return m_Data; }

            void Reset(int inX, int inY)
            {
                if(m_X != inX || m_Y != inY)
                {
                    deallocate(m_Data);

                    Init(inX, inY);
                }
            }

            void Rand(unsigned long long inSeed = 0)
            {
                // Create a pseudo-random number generator
                curandGenerator_t prng;
                curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

                // Set the seed for the random number generator using the system clock
                curandSetPseudoRandomGeneratorSeed(prng, inSeed != 0 ? inSeed : (unsigned long long) clock());

                // Fill the array with random numbers on the device
                curandGenerateUniform(prng, m_Data, m_X*m_Y);
            }

            MatrixGpu &operator=(const OperationGpu &inOperation);
            MatrixGpu &operator=(float inFill);
            MatrixGpu &operator=(const MatrixCpu& inMatrix);
            MatrixGpu &operator=(const MatrixGpu& inMatrix);

            OperationMatrixMultiply operator*(const MatrixGpu &inB) const;
            OperationMatrixAdd operator+(const MatrixGpu &inB) const;
            OperationMatrixSubstract operator-(const MatrixGpu &inB) const;

            OperationMatrixAggregate AbsMax(void) const;
            OperationMatrixAggregate AbsMin(void) const;
            OperationMatrixAggregate AbsSum(void) const;
            OperationBinaryAssociative Sum(void) const;
            OperationBinaryAssociative Min(void) const;
            OperationBinaryAssociative Max(void) const;
            OperationBinaryAssociative Multiply(void) const;

            OperationMatrixTransform operator^(const char *inType) const;

            MatrixGpu &operator^=(float inExponent);
            MatrixGpu &operator*=(float inVal);

        protected:
            void Init(int inX, int inY)
            {
                m_Data = allocate(inX*inY);
                m_X = inX;
                m_Y = inY;
            }

            int m_X;
            int m_Y;
            float *m_Data;
    };

    class OperationGpu
    {
        public:
            OperationGpu(void){}

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const = 0;

            virtual void GetResultSize(int &outX, int &outY) const = 0;

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
            OperationMatrixMultiply(const MatrixGpu& inA, const MatrixGpu& inB, bool inTransposeA = false, bool inTransposeB = false)
                : m_A(inA), m_B(inB), m_TransA(inTransposeA), m_TransB(inTransposeB)
            {
                int kA = !m_TransA ? m_A.getY() : m_A.getX();
                int kB = !m_TransB ? m_B.getX() : m_B.getY();
                assert(kA == kB);
            }

            virtual void GetResultSize(int &outX, int &outY) const
            {
                outX = !m_TransA ? m_A.getX() : m_A.getY();
                outY = !m_TransB ? m_B.getY() : m_B.getX();
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                int x = !m_TransA ? m_A.getX() : m_A.getY();
                int y = !m_TransB ? m_B.getY() : m_B.getX();
                int kA = !m_TransA ? m_A.getY() : m_A.getX();
                int kB = !m_TransB ? m_B.getX() : m_B.getY();

                //cout << "TA:" << m_TransA << ", TB:" << m_TransB << endl;
                assert(kA == kB);

                outMatrix.Reset(x, y);

                cublasSgemm(inHandle, !m_TransA ? CUBLAS_OP_N : CUBLAS_OP_T, !m_TransB ? CUBLAS_OP_N : CUBLAS_OP_T,
                        x, y, kA,
                        &m_One, m_A.getDataConst(), m_A.getX(), m_B.getDataConst(), m_B.getX(), &m_Zero, outMatrix.getDataConst(), x);

                //cublasSgemm(inHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                //        m_A.getX(), m_B.getY(), m_A.getY(),
                //        &m_One, m_A.getDataConst(), m_A.getX(), m_B.getDataConst(), m_B.getX(), &m_Zero, outMatrix.getDataConst(), m_A.getX());
            }

        protected:
            const MatrixGpu& m_A;
            const MatrixGpu& m_B;
            bool m_TransA;
            bool m_TransB;
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

            virtual void GetResultSize(int &outX, int &outY) const
            {
                outX = 1;
                outY = 1;
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                outMatrix.Reset(1, 1);


                static int ThreadsPerBlock = 512;

                int num = m_A.getX()*m_A.getY();

                int blocks = (num - 1) / ThreadsPerBlock + 1;

                float *tmp = allocate(blocks);

                parallelAssociativeOperator<<<blocks, ThreadsPerBlock, ThreadsPerBlock*sizeof(float)>>>(m_A.getDataConst(), num, m_Type, tmp);
                parallelAssociativeOperator<<<1, blocks, blocks*sizeof(float)>>>(tmp, blocks, m_Type, outMatrix.getData());
            }

        protected:
            const MatrixGpu& m_A;
            const EFunctionBinaryAssociative m_Type;
    };


    class OperationMatrixAdd : public OperationGpu
    {
        public:
            OperationMatrixAdd(const MatrixGpu& inA, const MatrixGpu& inB)
                : m_A(inA), m_B(inB)
            {
                assert (inA.getX() == inB.getX() && inA.getY() == inB.getY());
            }

            virtual void GetResultSize(int &outX, int &outY) const
            {
                outX = m_A.getX();
                outY = m_A.getY();
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                outMatrix.Reset(m_A.getX(), m_A.getY());

                cublasSgeam(inHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m_A.getX(), m_A.getY(),
                        &m_One, m_A.getDataConst(), m_A.getX(),
                        &m_One, m_B.getDataConst(), m_B.getX(),
                        outMatrix.getDataConst(), m_A.getX());
            }

        protected:
            const MatrixGpu& m_A;
            const MatrixGpu& m_B;
    };

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

    class OperationMatrixTransform : public OperationGpu
    {
        public:
            OperationMatrixTransform(const MatrixGpu& inA, const char *inType)
                : m_A(inA), m_Type(inType)
            {
                //only transpose now
                assert(inType[0] == 'T' && inType[1] == '\0');
            }

            virtual void GetResultSize(int &outX, int &outY) const
            {
                outX = m_A.getY();
                outY = m_A.getX();
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //static int ThreadsPerBlock = 32;
    
                //dim3 threadsPerBlock(ThreadsPerBlock, ThreadsPerBlock, 1);
    
                //dim3 threadsPerGrid((outMatrix.getX() - 1) / ThreadsPerBlock + 1, (outMatrix.getY() - 1) / ThreadsPerBlock + 1, 1);
    
                //transposeCoalesced<<<threadsPerGrid, threadsPerBlock>>>(outMatrix.getData(), m_A.getDataConst(), outMatrix.getX(), outMatrix.getY());
            }

            OperationMatrixMultiply operator*(const MatrixGpu &inB) const;

        protected:
            const MatrixGpu& m_A;
            bool m_Type;
    };

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

            virtual void GetResultSize(int &outX, int &outY) const
            {
                outX = 1;
                outY = 1;
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

    __global__ void applyFunction(float *outTarget, float *inSource, int N, EFunctionElementwise inType, float inParam1)
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
                }
            }
        }
    
    void funcElementwise(MatrixGpu &inOutMatrix, EFunctionElementwise inType, float inParam1 = 0.0f)
        {
            static int ThreadsPerBlock = 256;

            dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);

            int num = inOutMatrix.getX()*inOutMatrix.getY();

            dim3 threadsPerGrid((num - 1) / ThreadsPerBlock + 1, 1, 1);

            applyFunction<<<threadsPerGrid, threadsPerBlock>>>(inOutMatrix.getData(), inOutMatrix.getData(), num, inType, inParam1);
            //gpuErrCheck( cudaPeekAtLastError() );
            //gpuErrCheck( cudaDeviceSynchronize() );
        }

    class OperationMatrixApplyElementwise : public OperationGpu
    {
        public:
            OperationMatrixApplyElementwise(MatrixGpu& inA, EFunctionElementwise inType, float inParam1)
                : m_A(inA), m_Type(inType), m_Param1(inParam1)
            {
            }

            virtual void GetResultSize(int &outX, int &outY) const
            {
                outX = m_A.getX();
                outY = m_A.getY();
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                funcElementwise(m_A, m_Type, m_Param1);
            }

        protected:
            MatrixGpu& m_A;
            EFunctionElementwise m_Type;
            float m_Param1;
    };

    class OperationMatrixSubstract : public OperationGpu
    {
        public:
            OperationMatrixSubstract(const MatrixGpu& inA, const MatrixGpu& inB)
                : m_A(inA), m_B(inB)
            {
                assert (inA.getX() == inB.getX() && inA.getY() == inB.getY());
            }

            virtual void GetResultSize(int &outX, int &outY) const
            {
                outX = m_A.getX();
                outY = m_A.getY();
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                outMatrix.Reset(m_A.getX(), m_A.getY());

                cublasSgeam(inHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m_A.getX(), m_A.getY(),
                        &m_One, m_A.getDataConst(), m_A.getX(),
                        &m_MinusOne, m_B.getDataConst(), m_B.getX(),
                        outMatrix.getDataConst(), m_A.getX());
            }

        protected:
            const MatrixGpu& m_A;
            const MatrixGpu& m_B;
    };

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

    MatrixGpu::MatrixGpu(const MatrixGpu &inMatrix)
        {
            cout << "c";
            Init(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToDevice);
        }

    MatrixGpu::MatrixGpu(const MatrixCpu &inMatrix)
        {
            cout << "c";
            Init(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyHostToDevice);
        }

    MatrixGpu& MatrixGpu::operator=(const MatrixCpu &inMatrix)
        {
            Reset(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyHostToDevice);

            return *this;
        }
    MatrixGpu& MatrixGpu::operator=(const MatrixGpu &inMatrix)
        {
            Reset(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToDevice);

            return *this;
        }
    MatrixGpu& MatrixGpu::operator=(const OperationGpu &inOperation)
        {
            cublasHandle_t handle;
            cublasStatus_t stat;
        
            stat = cublasCreate(&handle);
            assert (stat == CUBLAS_STATUS_SUCCESS);

            inOperation.Execute(*this, handle);

            cublasDestroy(handle);

            return *this;
        }
    MatrixGpu &MatrixGpu::operator=(float inFill)
        {
            funcElementwise(*this, EFE_Fill, inFill);

            return *this;
        }

    MatrixGpu::MatrixGpu(const OperationGpu &inOperation)
        {
            int x, y;
            inOperation.GetResultSize(x, y);
            Init(x, y);

            cublasHandle_t handle;
            cublasStatus_t stat;
        
            stat = cublasCreate(&handle);
            assert (stat == CUBLAS_STATUS_SUCCESS);

            inOperation.Execute(*this, handle);
        }

    OperationMatrixMultiply MatrixGpu::operator*(const MatrixGpu &inB) const
        {
            return OperationMatrixMultiply(*this, inB);
        }

    OperationMatrixMultiply OperationMatrixTransform::operator*(const MatrixGpu &inB) const
        {
            return OperationMatrixMultiply(m_A, inB, m_Type, false);
        }

    OperationMatrixAdd MatrixGpu::operator+(const MatrixGpu &inB) const
        {
            return OperationMatrixAdd(*this, inB);
        }

    OperationMatrixSubstract MatrixGpu::operator-(const MatrixGpu &inB) const
        {
            return OperationMatrixSubstract(*this, inB);
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
    OperationMatrixTransform MatrixGpu::operator^(const char *inType) const
        {
            return OperationMatrixTransform(*this, inType);
        }
    OperationBinaryAssociative MatrixGpu::Sum(void) const
        {
            return OperationBinaryAssociative(*this, EFB_Plus);
        }

    OperationBinaryAssociative MatrixGpu::Multiply(void) const
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


}

#endif //MATRIX_H
