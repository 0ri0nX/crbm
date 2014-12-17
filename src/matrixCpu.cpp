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

    /*MatrixCpu &MatrixCpu::operator=(const MatrixGpu &inMatrix)
    {
        Reset(inMatrix.getX(), inMatrix.getY());
        assert(!inMatrix.isTrans());
        cudaMemcpy(m_Data, inMatrix.getDataConst(), inMatrix.getX()*inMatrix.getY()*sizeof(float), cudaMemcpyDeviceToHost);

        return *this;
    }*/

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

            void MatrixCpu::SampleY(t_index inColsNum, MatrixCpu &outSample) const
            {
                outSample.Reset(getX(), inColsNum);

                std::random_device randomDevice;
                std::uniform_int_distribution<int> dist(0, getY()-1);

                for(t_index i = 0; i < inColsNum; ++i)
                {
                    t_index randomCol = dist(randomDevice);

                    for(t_index j = 0; j < getX(); ++j)
                    {
                        outSample.set(j, i, get(i, randomCol));
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

    std::istream &MatrixCpu::Load(std::istream &inStream, bool inTransposed, const std::string &inCacheFileName)
    {
        std::string header;
        std::getline(inStream, header, '\n');
        //std::cout << "HEADER: [" << header << "]" << std::endl;
    
        const t_index lm = 6; //len(Matrix)
    
        if(header.substr(0, lm) == "Matrix")
        {
            std::stringstream hs(header.substr(lm, header.size() - 6));
    
            t_index version = 0;
            hs >> version;
    
            if(version == 1)//images ~ binary saved bytes => divide each value by 255
            {
                std::getline(inStream, header, '\n');
                std::stringstream hs(header);
                t_index x, y;
                hs >> x >> y;
    
                //assert (x >= 0 && y >= 0);
    
                //t_index sizeOfSavedt_index = 4;
                //assert(sizeof(int) == sizeOfSavedInt);
    
                //inStream.read((char*)&x, sizeOfSavedInt);
                //inStream.read((char*)&y, sizeOfSavedInt);
    
                std::cout << "x=" << x << ", y=" << y << std::endl;
    
                uint8_t d[y];
    
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

                if(!inTransposed)
                {
                    Reset(x, y, NULL, inCacheFileName);

                    for(t_index i = 0; i < x; ++i)
                    {
                        inStream.read((char*)d, y);
    
                        for(t_index j = 0; j < y; ++j)
                        {
                            //float dd = d[j]/255.0f;
                            //t_index idx = IDX2C(i, j, x);
                            //t_index mmax = x*y;
                            //if(idx >= mmax)
                            //{
                            //    //std::cout << "max int: " << numeric_limits<int>::max() << std::endl;
                            //    //std::cout << "max long: " << numeric_limits<long>::max() << std::endl;
                            //    //std::cout << "max ulong: " << numeric_limits<unsigned long>::max() << std::endl;
                            //    std::cout << "(" << (idx >= mmax) << ") " << idx << " >= " << x << " x " << y << " = " << mmax << ", ixj= " << i << " x " << j << std::endl;
                            //    assert(0);
                            //}

                            //m_Data[idx] = dd;
                            m_Data[IDX2C(i, j, x)] = d[j]/255.0f;
                        }

                        printProgress(i, x);
                    }
                }
                else
                {
                    Reset(y, x, NULL, inCacheFileName);
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
            else if (version == 2)//binary saved floats
            {
                std::getline(inStream, header, '\n');
                std::stringstream hs(header);
                t_index x, y;
                hs >> x >> y;
    
                //assert (x >= 0 && y >= 0);

                //t_index sizeOfSavedt_index = 4;
                //assert(sizeof(int) == sizeOfSavedInt);
    
                //inStream.read((char*)&x, sizeOfSavedInt);
                //inStream.read((char*)&y, sizeOfSavedInt);
    
                //std::cout << "x=" << x << ", y=" << y << std::endl;
    
                float d[y];
    
                t_index sizeOfSavedFloat = 4;
                assert(sizeof(float) == sizeOfSavedFloat);
    
                if(!inTransposed)
                {
                    Reset(x, y, NULL, inCacheFileName);
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
        }
        else//oldest-version
        {
            t_index x, y;
            std::stringstream hs(header);
            hs >> x >> y;
    
            //assert (x >= 0 && y >= 0);
    
            //cout << "x:" << x << "\ny:" << y << std::endl;
    
            if(!inTransposed)
            {
                Reset(x, y, NULL, inCacheFileName);
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
    
    
        return inStream;
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

}


