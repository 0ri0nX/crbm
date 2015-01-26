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
#include "utils.h"

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
        : m_X(0), m_Y(0), m_Data(NULL), m_CacheFileHandle(-1)
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
#ifndef SQUEEZE
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
#else
        assert(0);
#endif
    }
  
    void MatrixCpu::SampleCols(t_index inColsNum, MatrixCpu &outSample) const
    {
#ifndef SQUEEZE
        t_index randomCol[inColsNum];
  
        outSample.Reset(getX(), inColsNum);
  
        std::random_device randomDevice;
        std::uniform_int_distribution<int> dist(0, getY()-1);
  
        for(t_index i = 0; i < inColsNum; ++i)
        {
            randomCol[i] = dist(randomDevice);
            if(m_CacheFileHandle != -1)
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
  
            if(m_CacheFileHandle != -1)
            {
                madvise(getDataConst() + randomCol[i]*getX(), getX()*sizeof(float), MADV_NORMAL);
            }
        }
#else
        assert(0);
#endif
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

    /*std::istream &MatrixCpu::LoadHeader(std::istream &inStream, int &outVersion, t_index &outX, t_index &outY)
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
            else if (outVersion == 3)//only header -> binary saved floats are in other file
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
        
        std::cout << "version = " << outVersion << ", size = " << outX << " x " << outY << std::endl;

        return inStream;
    }*/

    /*std::istream &MatrixCpu::LoadBatch(std::istream &inStream, bool inTransposed, int inVersion, t_index x, t_index y, const std::string &inCacheFileName)
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
        else if (inVersion == 3)//binary saved floats in separate file that will be mapped into memory
        {
            if(m_CacheFileName == "")
            {
                std::string dataFile;
                lv(inStream, "DataFile", dataFile);
                Reset(y, x, NULL, dataFile);
            }
        }
        else
        {
            assert(0);
        }

        setCached(x*y, true);
    
        return inStream;
    }*/

    std::istream &MatrixCpu::Load(std::istream &inStream, bool inTransposed)
    {
        MatrixLoaderStream loader(&inStream);

        loader.LoadComplete(*this, inTransposed);
    }
    
    /*std::ostream &MatrixCpu::SaveHeader(std::ostream &outStream, t_index expectedRows, t_index expectedCols, int version)
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
    }*/

    std::ostream &MatrixCpu::Save(std::ostream &outStream) const
    {
        MatrixSaverStream saver(&outStream, 2);

        saver.SaveComplete(*this);
    }
/*
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
        else if(version == 2 || version == 3)
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
    }*/

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

    //"-" implies that all will be saved into one stream even if it is possible to split it
    MatrixSaverStream::MatrixSaverStream(std::ostream *inStream, int inVersion, const std::string &inFileNamePrefix)
        : m_MainStream(NULL)
          , m_SecondStream()
    {
        m_Step = 0;

        Reset(inStream, inVersion, inFileNamePrefix);
    }

    MatrixSaverStream::~MatrixSaverStream(void)
    {
        assert(m_Step == 0);
    }


    void MatrixSaverStream::Reset(std::ostream *inStream, int inVersion, const std::string &inFileNamePrefix)
    {
        assert(m_Step == 0);

        m_MainStream = inStream;
        m_Prefix = inFileNamePrefix;
        m_Version = inVersion;
    }

    void MatrixSaverStream::SaveComplete(const MatrixCpu &inMatrix)
    {
        PartSaveInit();
        PartSaveHeader(inMatrix.getX(), inMatrix.getY());
        PartSaveBatch(inMatrix);
        PartSaveFinish();
    }

    int MatrixSaverStream::getVersion(void) const
    {
        return m_Version;
    }

    const std::string &MatrixSaverStream::getPrefix(void) const
    {
        return m_Prefix;
    }

    int MatrixSaverStream::getStep() const
    {
        return m_Step;
    }

    void MatrixSaverStream::PartSaveInit()
    {
        if(getVersion() == -1)
        {
            return;
        }

        assert(m_Step == 0);

        if(getVersion() == 3)
        {
            if(m_Prefix != "-")
            {
                std::string bin = m_Prefix + ".bin";
                m_SecondStream.open(bin.c_str());
            }
        }

        m_Step = 1;
    }
    void MatrixSaverStream::PartSaveHeader(t_index inExpectedRows, t_index inExpectedCols)
    {
        if(getVersion() == -1)
        {
            return;
        }

        assert(m_Step == 1);

        if(getVersion() == 0)
        {
            (*m_MainStream) << inExpectedRows << " " << inExpectedCols << std::endl;
        }
        //else if(version == 1)
        //{
        //    std::stringstream e;
        //    e << "Matrix 1 version cannot be saved!" << std::endl;

        //    throw std::runtime_error(e.str());
        //}
        else if(getVersion() == 2)
        {
            (*m_MainStream) << "Matrix 2" << std::endl;
            (*m_MainStream) << inExpectedRows << " " << inExpectedCols << std::endl;
        }
        else if(getVersion() == 3)
        {
            (*m_MainStream) << "Matrix 3" << std::endl;
            (*m_MainStream) << inExpectedRows << " " << inExpectedCols << std::endl;
            if(m_Prefix == "-")
            {
                (*m_MainStream) << "DataFile -" << std::endl;
            }
            else
            {
                (*m_MainStream) << "DataFile" << " " << m_Prefix << ".bin" << std::endl;
            }
        }
        else
        {
            std::stringstream e;
            e << "Unknown version for matrix save, wanted [0,2,3] but got [" << getVersion() << "]" << std::endl;

            throw std::runtime_error(e.str());
        }

        m_Step = 2;
    }

    void MatrixSaverStream::PartSaveBatch(const MatrixCpu &inMatrix)
    {
        if(getVersion() == -1)
        {
            return;
        }

        assert(m_Step == 2 || m_Step == 3);

        if(getVersion() == 0)
        {
            for(t_index i = 0; i < inMatrix.m_X; ++i)
            {
                if(inMatrix.m_Y > 0)
                {
                    (*m_MainStream) << inMatrix.m_Data[IDX2C(i, 0, inMatrix.m_X)];
                }
    
                for(t_index j = 1; j < inMatrix.m_Y; ++j)
                {
                    (*m_MainStream) << " " << inMatrix.m_Data[IDX2C(i, j, inMatrix.m_X)];
                }
    
                (*m_MainStream) << std::endl;
            }
        }
        //else if(version == 1)
        //{
        //    std::stringstream e;
        //    e << "Matrix 1 version cannot be saved!" << std::endl;

        //    throw std::runtime_error(e.str());
        //}
        else if(getVersion() == 2 || getVersion() == 3)
        {
            std::ostream &outStream = (getVersion() == 3 && m_Prefix != "-") ? m_SecondStream : (*m_MainStream);

            t_index sizeOfSavedFloat = 4;
            assert(sizeof(float) == sizeOfSavedFloat);
            float d[inMatrix.m_Y];

            for(t_index i = 0; i < inMatrix.m_X; ++i)
            {
                for(t_index j = 0; j < inMatrix.m_Y; ++j)
                {
                    d[j] = inMatrix.m_Data[IDX2C(i, j, inMatrix.m_X)];
                }

                outStream.write((char*)d, inMatrix.m_Y*sizeOfSavedFloat);
            }
        }
        else
        {
            std::stringstream e;
            e << "Unknown version for matrix save, wanted [0-2] but got [" << getVersion() << "]" << std::endl;

            throw std::runtime_error(e.str());
        }

        m_Step = 3;
    }

    void MatrixSaverStream::PartSaveFinish(void)
    {
        if(getVersion() == -1)
        {
            return;
        }

        assert(m_Step == 3);

        if(getVersion() == 3)
        {
            if(m_Prefix != "-")
            {
                m_SecondStream.close();
            }
        }

        m_Step = 0;
    }

    void MatrixSaverFile::PartSaveInit(void)
    {
        if(getVersion() == -1)
        {
            return;
        }

        assert(getStep() == 0);

        std::string name = getPrefix() + ".dat";

        m_MainFileStream.open(name.c_str());

        Reset(&m_MainFileStream, getVersion(), getPrefix());

        MatrixSaverStream::PartSaveInit();

        assert(m_Step == 1);
    }

    void MatrixSaverFile::PartSaveFinish(void)
    {
        if(getVersion() == -1)
        {
            return;
        }

        assert(m_Step == 3);

        MatrixSaverStream::PartSaveFinish();

        m_MainFileStream.close();

        assert(m_Step == 0);
    }

    //version==-1 implies no saving
    MatrixSaverFile::MatrixSaverFile(const std::string &inFileNamePrefix, int inVersion)
    {
        m_Step = 0;
        Reset(NULL, inVersion, inFileNamePrefix);
    }

    void MatrixSaverFile::Reset(const std::string &inFileNamePrefix, int inVersion)
    {
        assert(getStep() == 0);

        Reset(NULL, inVersion, inFileNamePrefix);
    }
    void MatrixSaverFile::Reset(std::ostream *inStream, int inVersion, const std::string &inFileNamePrefix)
    {
        MatrixSaverStream::Reset(inStream, inVersion, inFileNamePrefix);
    }




    MatrixLoaderStream::MatrixLoaderStream(std::istream *inStream)
    {
        m_Step = 0;
        Reset(inStream);
    }
    MatrixLoaderStream::~MatrixLoaderStream(void)
    {
        assert(getStep() == 0);
    }

    void MatrixLoaderStream::Reset(std::istream *inStream)
    {
        assert(getStep() == 0);
        m_MainStream = inStream;
    }

    void MatrixLoaderStream::LoadComplete(MatrixCpu &outMatrix, bool inTransposed)
    {
        assert(getStep() == 0);

        PartLoadInit();
        t_index x, y;
        PartLoadHeader(x, y);
        PartLoadBatch(outMatrix, x, inTransposed);
        PartLoadFinish();

        assert(getStep() == 0);
    }

    int MatrixLoaderStream::getStep(void) const
    {
        return m_Step;
    }

    void MatrixLoaderStream::PartLoadInit(void)
    {
        assert(getStep() == 0);

        assert(m_MainStream != NULL);

        m_Step = 1;
    }

    void MatrixLoaderStream::PartLoadHeader(t_index &outX, t_index &outY)
    {
        assert(getStep() == 1);

        m_SecondFileName = "";

        std::string header;
        std::getline(*m_MainStream, header, '\n');
        //std::cout << "header1 [" << header << "]" << std::endl;
    
        const t_index lm = 6; //len("Matrix")
    
        if(header.substr(0, lm) == "Matrix")
        {
            std::stringstream hs(header.substr(lm, header.size() - lm));
    
            hs >> m_Version;
            if(m_Version == 1)//images ~ binary saved bytes => divide each value by 255
            {
                std::getline(*m_MainStream, header, '\n');
                std::stringstream hs(header);
                //hs.str(header);
                hs >> outX >> outY;
    
            }
            else if (m_Version == 2)//binary saved floats
            {
                std::getline(*m_MainStream, header, '\n');
                //std::cout << "header2 [" << header << "]" << std::endl;
                std::stringstream hs(header);
                //hs.str(header);
                //std::cout << "stream [" << hs.str() << "]" << std::endl;
                //std::cout << "uninited [" << outX << " x " << outY << "]" << std::endl;
                hs >> outX >> outY;
                //std::cout << "loaded [" << outX << " x " << outY << "]" << std::endl;
            }
            else if (m_Version == 3)//only header -> binary saved floats are in other file
            {
                std::getline(*m_MainStream, header, '\n');
                std::stringstream hs(header);
                //hs.str(header);
                hs >> outX >> outY;
                lv(*m_MainStream, "DataFile", m_SecondFileName);
            }
            else
            {
                std::stringstream e;
                e << "Matrix version [" << m_Version << "] is unknown!" << std::endl;

                throw std::runtime_error(e.str());
            }
        }
        else//oldest-version
        {
            m_Version = 0;
            std::stringstream hs(header);
            hs >> outX >> outY;
        }
        
        std::cout << "version = " << m_Version << ", size = " << outX << " x " << outY << std::endl;

        m_X = outX;
        m_Y = outY;

        assert(m_X > 0 && m_Y > 0);
        m_ReadX = 0;

        m_Step = 2;
    }

    //returns true while there is stil something to read
    bool MatrixLoaderStream::PartLoadBatch(MatrixCpu &outMatrix, t_index inMaxBatchSize, bool inTransposed)
    {
        assert(getStep() == 2 || getStep() == 3);

        //std::istream &MatrixCpu::LoadBatch(std::istream &inStream, bool inTransposed, int inVersion, t_index x, t_index y, const std::string &inCacheFileName)

        t_index batchSize = std::min(inMaxBatchSize, m_X - m_ReadX);

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

        //VERSION 0 ======================
        //floats in text format separated by spaces
        if(m_Version == 0)
        {
            if(!inTransposed)
            {
                outMatrix.Reset(batchSize, m_Y);

                for(t_index i = 0; i < batchSize; ++i)
                {
                    for(t_index j = 0; j < m_Y; ++j)
                    {
                        (*m_MainStream) >> outMatrix.m_Data[IDX2C(i, j, batchSize)];
                    }
                    printProgress(i, batchSize);
                }
            }
            else
            {
                outMatrix.Reset(m_Y, batchSize);
                for(t_index i = 0; i < batchSize; ++i)
                {
                    for(t_index j = 0; j < m_Y; ++j)
                    {
                        (*m_MainStream) >> outMatrix.m_Data[IDX2C(j, i, m_Y)];
                    }
                    printProgress(i, batchSize);
                }
            }
        }
        //VERSION 1 ======================
        //images ~ binary saved bytes => divide each value by 255
        else if(m_Version == 1)
        {
            uint8_t d[m_Y];
    
            if(!inTransposed)
            {
                outMatrix.Reset(batchSize, m_Y);

                for(t_index i = 0; i < batchSize; ++i)
                {
                    m_MainStream->read((char*)d, m_Y);
    
                    for(t_index j = 0; j < m_Y; ++j)
                    {
                        outMatrix.m_Data[IDX2C(i, j, batchSize)] = d[j]/255.0f;
                    }

                    printProgress(i, batchSize);
                }
            }
            else
            {
                outMatrix.Reset(m_Y, batchSize);

                for(t_index i = 0; i < batchSize; ++i)
                {
                    m_MainStream->read((char*)d, m_Y);
    
                    for(t_index j = 0; j < m_Y; ++j)
                    {
                        outMatrix.m_Data[IDX2C(j, i, m_Y)] = d[j]/255.0f;
                    }
                    printProgress(i, batchSize);
                }
            }
        }
        //VERSION 2 and 3 ======================
        //binary saved floats
        else if (m_Version == 2 || m_Version == 3)
        {
            //the only possibility to map file into memory is when loading the whole data and want the transposed version
            if(m_Version == 3 && m_SecondFileName != "-" && batchSize == m_X && inTransposed)
            {
                int fileHandle = open(m_SecondFileName.c_str(), O_RDWR);

                if (fileHandle == -1)
                {
                    throw std::runtime_error(std::string("Error opening file [" + m_SecondFileName + "] for writing"));
                }

                outMatrix.Reset(m_Y, m_X, NULL, fileHandle);
            }
            else
            {
                if(m_Version == 3 && m_SecondFileName != "-" && m_ReadX == 0)
                {
                    m_SecondStream.open(m_SecondFileName);

                    if(!m_SecondStream.is_open())
                    {
                        throw std::runtime_error(std::string("Error opening file [" + m_SecondFileName + "] for writing"));
                    }
                }

                std::istream &actStream = (m_SecondStream.is_open()) ? m_SecondStream : (*m_MainStream);
                float d[m_Y];
    
                t_index sizeOfSavedFloat = 4;
                assert(sizeof(float) == sizeOfSavedFloat);
    
                if(!inTransposed)
                {
                    outMatrix.Reset(batchSize, m_Y);

                    for(t_index i = 0; i < batchSize; ++i)
                    {
                        actStream.read((char*)d, m_Y*sizeOfSavedFloat);
    
                        for(t_index j = 0; j < m_Y; ++j)
                        {
                            outMatrix.m_Data[IDX2C(i, j, batchSize)] = d[j];
                        }
                        printProgress(i, batchSize);
                    }
                }
                else
                {
                    outMatrix.Reset(m_Y, batchSize);

                    for(t_index i = 0; i < batchSize; ++i)
                    {
                        actStream.read((char*)d, m_Y*sizeOfSavedFloat);
    
                        for(t_index j = 0; j < m_Y; ++j)
                        {
                            outMatrix.m_Data[IDX2C(j, i, m_Y)] = d[j];
                        }
                        printProgress(i, batchSize);
                    }
                }
            }
        }
        else
        {
            std::stringstream e;
            e << "Matrix version [" << m_Version << "] is unknown!" << std::endl;

            throw std::runtime_error(e.str());
        }

        m_ReadX += batchSize;

        m_Step = 3;

        return m_ReadX < m_X;
    }

    void MatrixLoaderStream::PartLoadFinish(void)
    {
        assert(getStep() == 3 || getStep() == 3);

        if(m_SecondStream.is_open())
        {
            m_SecondStream.close();
        }
        
        m_Step = 0;
    }

    MatrixLoaderFile::MatrixLoaderFile(const std::string &inFileName)
    {
        Reset(inFileName);
    }

    void MatrixLoaderFile::Reset(const std::string &inFileName)
    {
        assert(getStep() == 0);

        m_FileName = inFileName;
    }

    void MatrixLoaderFile::PartLoadInit(void)
    {
        assert(getStep() == 0);
        assert(m_FileName != "");

        m_MainFileStream.open(m_FileName.c_str());

        Reset(&m_MainFileStream);

        MatrixLoaderStream::PartLoadInit();

        assert(m_Step == 1);
    }

    void MatrixLoaderFile::PartLoadFinish(void)
    {
        assert(getStep() == 2 || getStep() == 3);

        MatrixLoaderStream::PartLoadFinish();

        assert(getStep() == 0);

        m_MainFileStream.close();
    }

    void MatrixLoaderFile::Reset(std::istream *inStream)
    {
        assert(getStep() == 0);

        MatrixLoaderStream::Reset(inStream);
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


