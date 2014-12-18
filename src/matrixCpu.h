#ifndef MATRIXCPU_H
#define MATRIXCPU_H

#include <limits>
#include <float.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <stdexcept>

//for memory map
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

namespace YAMATH
{
    typedef unsigned long t_index;
    class MatrixGpu;//forward

    class MatrixCpu//column-first layout
    {
        public:
            MatrixCpu(t_index inX = 1, t_index inY = 1, const float * inInit = NULL, std::string inCacheFileName = "") //column first order
                : m_X(0), m_Y(0), m_Data(NULL), m_CacheFileName(inCacheFileName), m_FileCache(-1)
            {
                Reset(inX, inY, inInit);
            }

            MatrixCpu(const MatrixGpu &inMatrix);
            MatrixCpu(const MatrixCpu &inMatrix);

//column-first order - ld is leading dimension size - #rows
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
            std::istream &Load(std::istream &inStream, bool inTransposed = false, const std::string &inCacheFileName = "");

            static std::ostream &SaveHeader(std::ostream &outStream, t_index expectedRows, t_index expectedCols, int version = 2);
            std::ostream &Save(std::ostream &outStream, bool addHeaderInfo = true, int version = 2) const;

            ~MatrixCpu(void)
            {
                DeallocateMemory();
            }

            t_index getX(void) const { return m_X; }
            t_index getY(void) const { return m_Y; }
            float* getDataConst(void) const { return m_Data; }
            float* getData(void) { return m_Data; }

            void Reshape(t_index inX, t_index inY)
            {
                assert(getX()*getY() == inX*inY);

                m_X = inX;
                m_Y = inY;
            }

            inline void set(t_index inX, t_index inY, float inValue)
            {
                m_Data[IDX2C(inX, inY, getX())] = inValue;
            }

            inline float get(t_index inX, t_index inY) const
            {
                return m_Data[IDX2C(inX, inY, getX())];
            }

            

            MatrixCpu SubMatrix(t_index inStartRow, t_index inStartCol, t_index inEndRow, t_index inEndCol) //start is inclusive, end is NON inclusive
            {
                assert(inEndRow <= getX());
                assert(inEndCol <= getY());
                
                //cout << "submatrix: " << inEndRow - inStartRow << "x" << inEndCol - inStartCol << endl;

                MatrixCpu m(inEndRow - inStartRow, inEndCol - inStartCol);

                for(t_index i = 0; i < m.getX(); ++i)
                {
                    for(t_index j = 0; j < m.getY(); ++j)
                    {
                        m.set(i, j, get(inStartRow + i, inStartCol + j));
                    }
                }

                return m;
            }

            void SubMatrixInsert(const MatrixCpu &inMatrix, t_index inStartRow, t_index inStartCol)//, t_index inEndRow, t_index inEndCol) //start is inclusive, end is NON inclusive
            {
                assert(inStartRow+inMatrix.getX() <= getX());
                assert(inStartCol+inMatrix.getY() <= getY());

                for(t_index i = 0; i < inMatrix.getX(); ++i)
                {
                    for(t_index j = 0; j < inMatrix.getY(); ++j)
                    {
                        set(inStartRow + i, inStartCol + j, inMatrix.get(i, j));
                    }
                }
            }

            //samples rows - not effective
            void Sample(t_index inRowsNum, MatrixCpu &outSample) const;

            //sampls columns - effective
            void SampleCols(t_index inColsNum, MatrixCpu &outSample) const;

            void AllocateMemory(t_index inDataSize, const std::string &inCacheFileName)
            {
                m_CacheFileName = inCacheFileName;

                if(m_CacheFileName != "")
                {
                    t_index size = inDataSize*sizeof(float);
                    //std::cout << "step 1" << std::endl;

                    m_FileCache = open(m_CacheFileName.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
                    if (m_FileCache == -1)
                    {
                        throw std::runtime_error("Error opening file for writing");
                    }

                    //std::cout << "step 2" << std::endl;
                    // Stretch the file size to the size of the (mmapped) array of ints
                    int result = lseek(m_FileCache, size-1, SEEK_SET);
                    if (result == -1)
                    {
                        close(m_FileCache);
                        throw std::runtime_error("Error calling lseek() to 'stretch' the file");
                    }

                    //std::cout << "step 3" << std::endl;
                    result = write(m_FileCache, "", 1);
                    if (result != 1)
                    {
                        close(m_FileCache);
                        throw std::runtime_error("Error writing last byte of the file");
                    }

                    //std::cout << "step 4" << std::endl;
                    m_Data = (float*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_FileCache, 0);

                    madvise(m_Data, size, MADV_RANDOM);

                    if (m_Data == MAP_FAILED)
                    {
                        close(m_FileCache);
                        throw std::runtime_error("Error mmapping the file");
                    }
                    //std::cout << "step 5" << std::endl;
                }
                else
                {
                    m_Data = new float [inDataSize];
                }
            }

            void DeallocateMemory(void)
            {
                if(m_CacheFileName != "")
                {
                    int result = munmap(m_Data, m_X*m_Y*sizeof(float));
                    if (result == -1)
                    {
                        perror("Error un-mmapping the file");
                    }
                    close(m_FileCache);
                }
                else
                {
                    delete [] m_Data;
                }
            }

            void Reset(t_index inX, t_index inY, const float * inInit = NULL, const std::string &inCacheFileName = "")
            {
                if(m_X*m_Y != inX*inY || m_CacheFileName != inCacheFileName)
                {
                    if(m_Data != NULL)
                    {
                        DeallocateMemory();
                    }

                    AllocateMemory(inX*inY, inCacheFileName);

                    assert(m_Data != NULL);
                }

                m_X = inX;
                m_Y = inY;

                if(inInit != NULL)
                {
                    memcpy(m_Data, inInit, inX*inY*sizeof(float));
                }
            }

            MatrixCpu &operator=(const MatrixGpu &inMatrix);
            MatrixCpu &operator=(const MatrixCpu &inMatrix);

        protected:

            /*void Init(t_index inX, t_index inY, const float *inInit = NULL)
            {
                assert (inX > 0 && inY > 0);

                m_Data = new float [inX*inY];
                m_X = inX;
                m_Y = inY;

                if(inInit != NULL)
                {
                    memcpy(m_Data, inInit, inX*inY*sizeof(float));
                }
            }*/

            t_index m_X;
            t_index m_Y;
            float *m_Data;
            std::string m_CacheFileName;
            int m_FileCache;
    };

}
void msgC(const char * inMsg, const YAMATH::MatrixCpu &x);




#endif //MATRIX_H

