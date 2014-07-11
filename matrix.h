#ifndef MATRIX_H
#define MATRIX_H

#include <cublas_v2.h>

namespace YAMATH
{
    float * allocate(int inNum)
    {
        float *ptr = NULL;
        cudaMalloc((void**) &ptr, inNum*sizeof(float));
        assert(ptr != NULL);
        return ptr;
    }

    void deallocate(float *inFloatArray)
    {
        cudaFree(inFloatArray);
    }

    class MatrixCpu;
    class MatrixGpu;
    class OperationGpu;
    class OperationMatrixMultiply;

    class MatrixCpu//column-first layout
    {
        public:
            MatrixCpu(int x = 1, int y = 1)
                : m_X(x), m_Y(y), m_Data(NULL)
            {
                assert (x > 0 && y > 0);
                m_Data = new float [m_X*m_Y];
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

            void Reset(int inX, int inY)
            {
                if(m_X != inX || m_Y != inY)
                {
                    delete [] m_Data;
                    Init(inX, inY);
                }
            }

        protected:
            void Init(int inX, int inY)
            {
                m_Data = new float [inX*inY];
                m_X = inX;
                m_Y = inY;
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
            }

//column-first order - ld is leading dimension size - #rows
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

            MatrixGpu(const MatrixGpu &inMatrix);
            MatrixGpu(const MatrixCpu &inMatrix);

            ~MatrixGpu(void)
            {
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

            MatrixGpu &operator=(const OperationGpu &inOperation);

            OperationMatrixMultiply operator*(const MatrixGpu &inB) const;

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

    };

    const float OperationGpu::m_Zero = 0.0f;
    const float OperationGpu::m_One = 1.0f;

    class OperationMatrixMultiply : public OperationGpu
    {
        public:
            OperationMatrixMultiply(const MatrixGpu& inA, const MatrixGpu& inB)
                : m_A(inA), m_B(inB)
            {
                assert (inA.getY() == inB.getX());
            }

            virtual void GetResultSize(int &outX, int &outY) const
            {
                outX = m_A.getX();
                outY = m_B.getY();
            }

            virtual void Execute(MatrixGpu &outMatrix, cublasHandle_t inHandle) const
            {
                //assert(outMatrix.this != m_A.this && outMatrix.this != m_B.this);

                outMatrix.Reset(m_A.getX(), m_B.getY());

                cublasSgemm(inHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m_A.getX(), m_B.getY(), m_A.getY(),
                        &m_One, m_A.getDataConst(), m_A.getX(), m_B.getDataConst(), m_B.getX(), &m_Zero, outMatrix.getDataConst(), m_A.getX());
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
            Init(inMatrix.getX(), inMatrix.getY());
            memcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float));
        }

    MatrixGpu::MatrixGpu(const MatrixGpu &inMatrix)
        {
            Init(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToDevice);
        }

    MatrixGpu::MatrixGpu(const MatrixCpu &inMatrix)
        {
            Init(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyHostToDevice);
        }

    MatrixGpu& MatrixGpu::operator=(const OperationGpu &inOperation)
        {
            cublasHandle_t handle;
            cublasStatus_t stat;
        
            stat = cublasCreate(&handle);
            assert (stat == CUBLAS_STATUS_SUCCESS);

            inOperation.Execute(*this, handle);

            return *this;
        }

    OperationMatrixMultiply MatrixGpu::operator*(const MatrixGpu &inB) const
        {
            return OperationMatrixMultiply(*this, inB);
        }

}

/*
    template<int ThreadsPerBlock>
    struct DeviceGPU
    {
        static const int m_MaxSharedMemPerBlock = 16*1024;

        //returns name
        static const char* GetName(void)
        {
            return "gpu";
        }

        static void randomSeed(unsigned int inSeed);
        static void vectorRandom(float *outVector, int inNum, float inMin, float inMax);
        static void vectorSet(float *outVector, int inNum, float inValue);
        //if inOutputAccu >= 1 then inOutput = 1 and inOutputAccu=0
        static void forwardSpike(float *inOutputAccu, float* inOutput, int inNum);
        static void vectorThreshold(float *inInput, int inNum, float inThreshold);
        //inOutput += inInput x inWeights
        static void vectorMultMatrix(float *inOutput, float* const inInput, float * const inWeights, int inNumInp, int inNumOut);
        //A += B
        static void vectorAdd(float *inA, float * const inB, int inNum);
        static void getVector(float *outHost, float *inDevice, int inNum);
        static float * allocate(int inNum);
        static void deallocate(float *inFloatArray);
    };

}//namespace ComputingDevice

//include definition cuh file
#include"deviceGPU.hpp"


#endif //DEVICEGPU_H

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>

#include "deviceGPU.h"

namespace ComputingDevice
{
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

    inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
    {
       if (code != cudaSuccess) 
       {
          fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
          if (abort) exit(code);
       }
    }

    //instantiation
    //template class DeviceGPU<64>;
    //template class DeviceGPU<128>;
    //template class DeviceGPU<256>;
    //template class DeviceGPU<512>;


    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::randomSeed(unsigned int inSeed)
        {
            srand(inSeed);
        }

    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::vectorRandom(float *outVector, int inNum, float inMin, float inMax)
        {
            const float w = inMax - inMin;

            float tmpBuffer[inNum];

            for(int i = 0; i < inNum; ++i)
            {
                tmpBuffer[i] = inMin + w * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
            gpuErrCheck(cudaMemcpy(outVector, tmpBuffer, sizeof(float)*inNum, cudaMemcpyHostToDevice));
        }

        __global__ void vectorSetKernel(float *outVector, int inNum, float inValue)
        {
            int i = blockDim.x*blockIdx.x + threadIdx.x;
            if(i < inNum)
            {
                outVector[i] = inValue;
            }
        }
    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::vectorSet(float *outVector, int inNum, float inValue)
        {
            dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);
            dim3 threadsPerGrid((inNum - 1) / ThreadsPerBlock + 1, 1, 1);

            vectorSetKernel<<<threadsPerGrid, threadsPerBlock>>>(outVector, inNum, inValue);
            gpuErrCheck( cudaPeekAtLastError() );
            gpuErrCheck( cudaDeviceSynchronize() );
        }
        
        __global__ void forwardSpikeKernel(float *inOutputAccu, float* inOutput, int inNum)
        {
            int i = blockDim.x*blockIdx.x + threadIdx.x;
            if(i < inNum)
            {
                if(inOutputAccu[i] >= 1.0f)
                {
                    inOutput[i] = 1.0f;
                    inOutputAccu[i] = 0.0f;
                }
                else
                {
                    inOutput[i] = 0.0f;
                    if(inOutputAccu[i] < 0.0f)
                    {
                        inOutputAccu[i] = 0.0f;
                    }
                }
            }
        }
        //if inOutputAccu >= 1 then inOutput = 1 and inOutputAccu=0
    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::forwardSpike(float *inOutputAccu, float* inOutput, int inNum)
        {
            dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);
            dim3 threadsPerGrid((inNum - 1) / ThreadsPerBlock + 1, 1, 1);

            forwardSpikeKernel<<<threadsPerGrid, threadsPerBlock>>>(inOutputAccu, inOutput, inNum);
            gpuErrCheck( cudaPeekAtLastError() );
            gpuErrCheck( cudaDeviceSynchronize() );
        }
        
        __global__ void vectorThresholdKernel(float *inInput, int inNum, float inThreshold)
        {
            int i = blockDim.x*blockIdx.x + threadIdx.x;
            if(i < inNum)
            {
                inInput[i] = inInput[i] > inThreshold ? 1.0f : 0.0f;
            }
        }
    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::vectorThreshold(float *inInput, int inNum, float inThreshold)
        {
            dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);
            dim3 threadsPerGrid((inNum - 1) / ThreadsPerBlock + 1, 1, 1);

            vectorThresholdKernel<<<threadsPerGrid, threadsPerBlock>>>(inInput, inNum, inThreshold);
            gpuErrCheck( cudaPeekAtLastError() );
            gpuErrCheck( cudaDeviceSynchronize() );
        }
        
        __global__ void vectorMultMatrixKernelSimple(float *inOutput, float* const inInput, float * const inWeights, int inNumInp, int inNumOut)
        {
            int i = blockDim.x*blockIdx.x + threadIdx.x;

            //calculate only when neccessary
            if(i < inNumOut)
            {
                float sum = 0.0f;

                for(int j = 0; j < inNumInp; ++j)
                {
                    sum += inInput[j] * inWeights[inNumOut*j + i];
                }
                
                inOutput[i] += sum;
            }
        }

        //this is not faster than simple kernel above
        __global__ void vectorMultMatrixKernel(float *inOutput, float* const inInput, float * const inWeights, int inNumInp, int inNumOut, int inMaxSharedFloatPerBlock)
        {
            extern __shared__ float vec[];

            int i = blockDim.x*blockIdx.x + threadIdx.x;

            //if vector is greater then shared-mem size then split it to parts
            int parts = (inNumInp -1) / inMaxSharedFloatPerBlock + 1;

            float sum = 0.0f;

            for(int part = 0; part < parts; ++part)
            {
                int realIndexStart = part*inMaxSharedFloatPerBlock;

                //copy to shared mem
                if(realIndexStart + threadIdx.x < inNumInp && threadIdx.x < inMaxSharedFloatPerBlock)
                {
                    vec[threadIdx.x] = inInput[realIndexStart + threadIdx.x];
                }
                //sync shared memory
                __syncthreads();

                //calculate only when neccessary
                if(i < inNumOut)
                {
                    for(int j = 0; j < inMaxSharedFloatPerBlock && realIndexStart + j < inNumInp; ++j)
                    {
                        sum += vec[j] * inWeights[(realIndexStart + j)*inNumOut + i];//memory coalescing
                    }
                }

                //sync partial multiplication
                __syncthreads();
                
            }

            //set only when neccessary
            if(i < inNumOut)
            {
                inOutput[i] += sum;
            }
        }
        //inOutput += inInput x inWeights
    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::vectorMultMatrix(float *inOutput, float* const inInput, float * const inWeights, int inNumInp, int inNumOut)
        {
            dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);
            dim3 threadsPerGrid((inNumOut - 1) / ThreadsPerBlock + 1, 1, 1);

            //simple-version - it is actually faster then the shared one
            if(1)
            {
                vectorMultMatrixKernelSimple<<<threadsPerGrid, threadsPerBlock>>>(inOutput, inInput, inWeights, inNumInp, inNumOut);
            }
            else//shared-memory version
            {
                //allocate enough shared memory per block
                int sharedFloatMemSize = m_MaxSharedMemPerBlock/sizeof(float) < inNumInp ? m_MaxSharedMemPerBlock/sizeof(float) : inNumInp;

                //but do not allocate more than we can use
                if(ThreadsPerBlock < sharedFloatMemSize)
                {
                    sharedFloatMemSize = ThreadsPerBlock;
                }

                vectorMultMatrixKernel<<<threadsPerGrid, threadsPerBlock, sharedFloatMemSize*sizeof(float)>>>(inOutput, inInput, inWeights, inNumInp, inNumOut, sharedFloatMemSize);
            }
            gpuErrCheck( cudaPeekAtLastError() );
            gpuErrCheck( cudaDeviceSynchronize() );
        }
        

        __global__ void vectorAddKernel(float *inA, float * const inB, int inNum)
        {
            int i = blockDim.x*blockIdx.x + threadIdx.x;
            if(i < inNum)
            {
                inA[i] += inB[i];
            }
        }
        //A += B
    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::vectorAdd(float *inA, float * const inB, int inNum)
        {
            dim3 threadsPerBlock(ThreadsPerBlock, 1, 1);
            dim3 threadsPerGrid((inNum - 1) / ThreadsPerBlock + 1, 1, 1);

            vectorAddKernel<<<threadsPerGrid, threadsPerBlock>>>(inA, inB, inNum);
            gpuErrCheck( cudaPeekAtLastError() );
            gpuErrCheck( cudaDeviceSynchronize() );
        }


    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::getVector(float *outHost, float *inDevice, int inNum)
        {
            gpuErrCheck( cudaMemcpy(outHost, inDevice, sizeof(float)*inNum, cudaMemcpyDeviceToHost));
        }


    template<int ThreadsPerBlock>
        float * DeviceGPU<ThreadsPerBlock>::allocate(int inNum)
        {
            float *ptr = NULL;
            cudaMalloc((void**) &ptr, inNum*sizeof(float));
            assert(ptr != NULL);
            return ptr;
        }

    template<int ThreadsPerBlock>
        void DeviceGPU<ThreadsPerBlock>::deallocate(float *inFloatArray)
        {
            cudaFree(inFloatArray);
        }

}//namespace ComputingDevice
*/

#endif //MATRIX_H
