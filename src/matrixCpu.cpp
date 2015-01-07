#include <limits>
#include <float.h>
#include <cstdlib>
#include <ctime>
#include <iomanip>
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

#include <random>

#include "matrixCpu.h"

#include <cblas.h>
namespace YAMATH
{
    /*MatrixCpu::MatrixCpu(const MatrixGpu &inMatrix)
        : m_X(0), m_Y(0), m_Data(NULL), m_CacheFileName(""), m_FileCache(-1)
        {
            Reset(inMatrix.getX(), inMatrix.getY());
            cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToHost);
        }*/

    MatrixCpu::MatrixCpu(const MatrixCpu &inMatrix)
        : m_X(0), m_Y(0), m_Data(NULL), m_CacheFileName(""), m_FileCache(-1)
        {
            Reset(inMatrix.getX(), inMatrix.getY(), inMatrix.getDataConst());
        }

    MatrixCpu &MatrixCpu::operator=(float inVal)
    {
        for(t_index i = 0; i < getX()*getY(); ++i)
        {
            getData()[i] = inVal;
        }
    }

    MatrixCpu &MatrixCpu::operator=(const MatrixCpu &inMatrix)
    {
        if(this != &inMatrix)
        {
            Reset(inMatrix.getX(), inMatrix.getY(), inMatrix.getDataConst());
        }
  
        return *this;
    }
  
    void MatrixCpu::Sample(t_index inRowsNum, MatrixCpu &outSample) const
    {
        outSample.Reset(inRowsNum, getY());
  
        std::random_device randomDevice;
        std::uniform_int_distribution<int> dist(0, getX()-1);
  
        for(t_index i = 0; i < inRowsNum; ++i)
        {
            t_index randomRow = dist(randomDevice);
  
            for(t_index j = 0; j < getY(); ++j)
            {
                outSample.set(i, j, get(randomRow, j));
            }
        }
    }
  
    void MatrixCpu::SampleCols(t_index inColsNum, MatrixCpu &outSample) const
    {
        t_index randomCol[inColsNum];
  
        outSample.Reset(getX(), inColsNum);
  
        std::random_device randomDevice;
        std::uniform_int_distribution<int> dist(0, getY()-1);
  
        for(t_index i = 0; i < inColsNum; ++i)
        {
            randomCol[i] = dist(randomDevice);
            if(m_CacheFileName != "")
            {
                madvise(getDataConst() + randomCol[i]*getX(), getX()*sizeof(float), MADV_WILLNEED);
            }
        }
  
        for(t_index i = 0; i < inColsNum; ++i)
        {
            memcpy(outSample.getData() + i*getX(), getDataConst() + randomCol[i]*getX(), getX()*sizeof(float));
            for(t_index j = 0; j < getX(); ++j)
            {
                float* res = outSample.getData() + i*getX() + j;

                if(isnan(*res) || isinf(*res))
                {
                    std::cout << "NaN: " << randomCol[i] << " , " << j << std::endl;
                }
            }
  
            if(m_CacheFileName != "")
            {
                madvise(getDataConst() + randomCol[i]*getX(), getX()*sizeof(float), MADV_NORMAL);
            }
        }
    }

    void printProgress(int act, int max)
    {
        //if(act > 1)
        //{
        //    if((100*(act-1))/max < (100*act)/max)
        //    {
        //        std::cout << " " << ((100*act)/max) << "%" << "\r" << std::flush;
        //    }
        //}
        if(act % 100 == 0)
        {
            std::cout << act << " " << std::setprecision(2) << std::fixed << (float(100*act)/max) << "%" << "\r" << std::scientific << std::flush;
        }
    }

    std::istream &MatrixCpu::LoadHeader(std::istream &inStream, int &outVersion, t_index &outX, t_index &outY)
    {
        std::string header;
        std::getline(inStream, header, '\n');
    
        const t_index lm = 6; //len("Matrix")
    
        if(header.substr(0, lm) == "Matrix")
        {
            std::stringstream hs(header.substr(lm, header.size() - lm));
    
            hs >> outVersion;
            if(outVersion == 1)//images ~ binary saved bytes => divide each value by 255
            {
                std::getline(inStream, header, '\n');
                std::stringstream hs(header);
                hs >> outX >> outY;
    
            }
            else if (outVersion == 2)//binary saved floats
            {
                std::getline(inStream, header, '\n');
                std::stringstream hs(header);
                hs >> outX >> outY;
            }
        }
        else//oldest-version
        {
            outVersion = 0;
            std::stringstream hs(header);
            hs >> outX >> outY;
        }
        
        std::cout << "version= " << outVersion << ", size= " << outX << " x " << outY << std::endl;

        return inStream;
    }

    std::istream &MatrixCpu::LoadBatch(std::istream &inStream, bool inTransposed, int inVersion, t_index x, t_index y, const std::string &inCacheFileName)
    {
        if(inVersion == 0)
        {
            if(!inTransposed)
            {
                Reset(x, y, NULL, inCacheFileName);
                if(isCached(x*y))
                {
                    std::cout << "Using cache in the file: [" << inCacheFileName << "]" << std::endl;
                    return inStream;
                }
                for(t_index i = 0; i < x; ++i)
                {
                    for(t_index j = 0; j < y; ++j)
                    {
                        inStream >> m_Data[IDX2C(i, j, x)];
                    }
                    printProgress(i, x);
                }
            }
            else
            {
                Reset(y, x, NULL, inCacheFileName);
                if(isCached(x*y))
                {
                    std::cout << "Using cache in the file: [" << inCacheFileName << "]" << std::endl;
                    return inStream;
                }
                for(t_index i = 0; i < x; ++i)
                {
                    for(t_index j = 0; j < y; ++j)
                    {
                        inStream >> m_Data[IDX2C(j, i, y)];
                    }
                    printProgress(i, x);
                }
            }
        }
        else if(inVersion == 1)//images ~ binary saved bytes => divide each value by 255
        {
            uint8_t d[y];
    
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

            if(!inTransposed)
            {
                Reset(x, y, NULL, inCacheFileName);

                if(isCached(x*y))
                {
                    std::cout << "Using cache in the file: [" << inCacheFileName << "]" << std::endl;
                    return inStream;
                }
                for(t_index i = 0; i < x; ++i)
                {
                    inStream.read((char*)d, y);
    
                    for(t_index j = 0; j < y; ++j)
                    {
                        m_Data[IDX2C(i, j, x)] = d[j]/255.0f;
                    }

                    printProgress(i, x);
                }
            }
            else
            {
                Reset(y, x, NULL, inCacheFileName);
                if(isCached(x*y))
                {
                    std::cout << "Using cache in the file: [" << inCacheFileName << "]" << std::endl;
                    return inStream;
                }
                for(t_index i = 0; i < x; ++i)
                {
                    inStream.read((char*)d, y);
    
                    for(t_index j = 0; j < y; ++j)
                    {
                        m_Data[IDX2C(j, i, y)] = d[j]/255.0f;
                    }
                    printProgress(i, x);
                }
            }
        }
        else if (inVersion == 2)//binary saved floats
        {
            float d[y];
    
            t_index sizeOfSavedFloat = 4;
            assert(sizeof(float) == sizeOfSavedFloat);
    
            if(!inTransposed)
            {
                Reset(x, y, NULL, inCacheFileName);
                if(isCached(x*y))
                {
                    std::cout << "Using cache in the file: [" << inCacheFileName << "]" << std::endl;
                    return inStream;
                }
                for(t_index i = 0; i < x; ++i)
                {
                    inStream.read((char*)d, y*sizeOfSavedFloat);
    
                    for(t_index j = 0; j < y; ++j)
                    {
                        m_Data[IDX2C(i, j, x)] = d[j];
                    }
                    printProgress(i, x);
                }
            }
            else
            {
                Reset(y, x, NULL, inCacheFileName);
                if(isCached(x*y))
                {
                    std::cout << "Using cache in the file: [" << inCacheFileName << "]" << std::endl;
                    return inStream;
                }
                for(t_index i = 0; i < x; ++i)
                {
                    inStream.read((char*)d, y*sizeOfSavedFloat);
    
                    for(t_index j = 0; j < y; ++j)
                    {
                        m_Data[IDX2C(j, i, y)] = d[j];
                    }
                    printProgress(i, x);
                }
            }
        }
        else
        {
            assert(0);
        }

        setCached(x*y, true);
    
        return inStream;
    }

    std::istream &MatrixCpu::Load(std::istream &inStream, bool inTransposed, const std::string &inCacheFileName)
    {
        t_index x, y;
        int version;

        LoadHeader(inStream, version, x, y);
        LoadBatch (inStream, inTransposed, version, x, y, inCacheFileName);
    }
    
    std::ostream &MatrixCpu::SaveHeader(std::ostream &outStream, t_index expectedRows, t_index expectedCols, int version)
    {
        if(version == 0)
        {
            outStream << expectedRows << " " << expectedCols << std::endl;
        }
        else if(version == 1)
        {
            std::stringstream e;
            e << "Matrix 1 version cannot be saved!" << std::endl;

            throw std::runtime_error(e.str());
        }
        else if(version == 2)
        {
            outStream << "Matrix 2" << std::endl;
            outStream << expectedRows << " " << expectedCols << std::endl;
        }
        else
        {
            std::stringstream e;
            e << "Unknown version for matrix save, wanted [0 or 2] but got [" << version << "]" << std::endl;

            throw std::runtime_error(e.str());
        }

        return outStream;
    }
    std::ostream &MatrixCpu::Save(std::ostream &outStream, bool addHeaderInfo, int version) const
    {
        if(addHeaderInfo)
        {
            SaveHeader(outStream, getX(), getY(), version);
        }

        if(version == 0)
        {
            for(t_index i = 0; i < m_X; ++i)
            {
                if(m_Y > 0)
                {
                    outStream << m_Data[IDX2C(i, 0, m_X)];
                }
    
                for(t_index j = 1; j < m_Y; ++j)
                {
                    outStream << " " << m_Data[IDX2C(i, j, m_X)];
                }
    
                outStream << std::endl;
            }
        }
        else if(version == 1)
        {
            std::stringstream e;
            e << "Matrix 1 version cannot be saved!" << std::endl;

            throw std::runtime_error(e.str());
        }
        else if(version == 2)
        {
            //t_index sizeOfSavedt_index = 4, x = m_X, y = m_Y;
            //assert(sizeof(int) == sizeOfSavedInt);

            //outStream.write((char*)&x, sizeOfSavedInt);
            //outStream.write((char*)&y, sizeOfSavedInt);

            t_index sizeOfSavedFloat = 4;
            assert(sizeof(float) == sizeOfSavedFloat);
            float d[m_Y];

            for(t_index i = 0; i < m_X; ++i)
            {
                for(t_index j = 0; j < m_Y; ++j)
                {
                    d[j] = m_Data[IDX2C(i, j, m_X)];
                }

                outStream.write((char*)d, m_Y*sizeOfSavedFloat);
            }
        }
        else
        {
            std::stringstream e;
            e << "Unknown version for matrix save, wanted [0-2] but got [" << version << "]" << std::endl;

            throw std::runtime_error(e.str());
        }


        if(0)//raw data
        {
            outStream << "raw:";
            for(t_index j = 0; j < m_Y*m_Y; ++j)
            {
                outStream << " " << m_Data[j];
            }
            outStream << std::endl;
        }
    
        return outStream;
    }

    void MatrixCpu::RandNormal(float inMean, float inStdDev, unsigned long long inSeed)//normal randomess, mean 0.0f standard deviation 1.0f
    {
        //TODO: inSeed != 0 ? inSeed : (unsigned long long) clock()
        std::random_device randomDevice;
        std::normal_distribution<float> dist(inMean, inStdDev);
  
        for(t_index i = 0; i < getX()*getY(); ++i)
        {
            getData()[i] = dist(randomDevice);
        }
    }

    MatrixCpu &MatrixCpu::operator*=(const MatrixCpu &inB)//elementwise multiplication!
    {
        for(t_index i = 0; i < getX()*getY(); ++i)
        {
            getData()[i] *= inB.getDataConst()[i];
        }
    }

    MatrixCpu Mult(const MatrixCpu &inA, const MatrixCpu &inB, bool transposedA, bool transposedB)//matrix multiplication!
    {
        t_index x = !transposedA ? inA.getX() : inA.getY();
        t_index y = !transposedB ? inB.getY() : inB.getX();
        t_index kA = !transposedA ? inA.getY() : inA.getX();
        t_index kB = !transposedB ? inB.getX() : inB.getY();

        //cout << "TA:" << inA.isTrans() << ", TB:" << inB.isTrans() << endl;
        assert(kA == kB);

        MatrixCpu outMatrix(x, y);

        //void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB
        //         , const int M, const int N, const int K
        //         , const float alpha, const float *A, const int lda
        //         , const float *B, const int ldb,
        //         const float beta, float *C, const int ldc);

        cblas_sgemm(CblasColMajor, !transposedA ? CblasNoTrans : CblasTrans, !transposedB ? CblasNoTrans : CblasTrans
                , x, y, kA
                , 1.0f, inA.getDataConst(), inA.getX()
                , inB.getDataConst(), inB.getX()
                , 0.0f, outMatrix.getDataConst(), x);

        return outMatrix;
    }
}

void msgC(const char * inMsg, const YAMATH::MatrixCpu &x)
{
    int n = x.getX()*x.getY();
    if(n > 400)
    {
        std::cout << inMsg << ": " << x.getX() << " x " << x.getY()
             << "[ " << (x.getDataConst()[0]) << ", " << (x.getDataConst()[1]) << " ... " << (x.getDataConst()[n-2]) << ", " << (x.getDataConst()[n-1]) << " ]" << std::endl;
    }
    else if(n == 1)
    {
        std::cout  << inMsg << ": " << x.getX() << " x " << x.getY() << ":[" << x.getDataConst()[0] << "]" << std::flush;
    }
    else
    {
        std::cout  << inMsg << ": " << x.getX() << " x " << x.getY() << ":" << std::endl;
        x.Save(std::cout);
        std::cout << std::endl;
    }
}


