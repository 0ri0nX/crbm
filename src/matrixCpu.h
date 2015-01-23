#ifndef MATRIXCPU_H
#define MATRIXCPU_H

#include <limits>
#include <float.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
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

    class MatrixCpu;//forward

    class MatrixSaverStream
    {
        public:
            MatrixSaverStream(std::ostream *inStream = NULL, int inVersion = 2, const std::string &inFileNamePrefix = "-");//"-" implies that all will be saved into one stream even if it is possible to split it
            ~MatrixSaverStream(void);

            virtual void        Reset(std::ostream *inStream, int inVersion = 2, const std::string &inFileNamePrefix = "-");

            void                SaveComplete(const MatrixCpu &inMatrix);

            int                 getVersion(void) const;
            const std::string   &getPrefix(void) const;
            int                 getStep(void) const;

            virtual void        PartSaveInit(void);
            virtual void        PartSaveHeader(t_index inExpectedRows, t_index inExpectedCols);
            virtual void        PartSaveBatch(const MatrixCpu &inMatrix);
            virtual void        PartSaveFinish(void);

        protected:

            std::ostream        *m_MainStream;
            std::ofstream       m_SecondStream;
            int                 m_Version;
            std::string         m_Prefix;
            int                 m_Step;
    };

    class MatrixSaverFile : public MatrixSaverStream
    {
        public:
            MatrixSaverFile(const std::string &inFileNamePrefix = "", int inVersion = 2);//version==-1 implies no saving

            void                Reset(const std::string &inFileNamePrefix, int inVersion = 2);

            virtual void        PartSaveInit(void);
            virtual void        PartSaveFinish(void);

        protected:
            virtual void        Reset(std::ostream *inStream, int inVersion, const std::string &inFileNamePrefix);

            std::ofstream       m_MainFileStream;
    };

    class MatrixLoaderStream
    {
        public:
            MatrixLoaderStream(std::istream *inStream = NULL);
            ~MatrixLoaderStream(void);

            virtual void        Reset(std::istream *inStream);

            void                LoadComplete(MatrixCpu &outMatrix, bool inTransposed = false);

            int                 getStep(void) const;

            virtual void        PartLoadInit(void);
            virtual void        PartLoadHeader(t_index &outRows, t_index &outCols);
            virtual bool        PartLoadBatch(MatrixCpu &outMatrix, t_index inMaxBatchSize, bool inTransposed = false);//returns true while there is stil something to read
            virtual void        PartLoadFinish(void);

        protected:

            std::istream        *m_MainStream;
            std::ifstream       m_SecondStream;
            int                 m_Version;
            std::string         m_SecondFileName;
            int                 m_Step;
            t_index             m_X;
            t_index             m_Y;
            t_index             m_ReadX;
    };

    class MatrixLoaderFile : public MatrixLoaderStream
    {
        public:
            MatrixLoaderFile(const std::string &inFileName = "");

            void                Reset(const std::string &inFileName);

            virtual void        PartLoadInit(void);
            virtual void        PartLoadFinish(void);

        protected:
            virtual void        Reset(std::istream *inStream);

            std::ifstream       m_MainFileStream;
            std::string         m_FileName;
    };

    class MatrixCpu//column-first layout
    {
        public:
            MatrixCpu(t_index inX = 1, t_index inY = 1, const float * inInit = NULL) //column first order
                : m_X(0), m_Y(0), m_Data(NULL), m_CacheFileHandle(-1)
            {
                Reset(inX, inY, inInit);
            }

            MatrixCpu(const MatrixGpu &inMatrix);
            MatrixCpu(const MatrixCpu &inMatrix);

//column-first order - ld is leading dimension size - #rows
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

            std::ostream &Save(std::ostream &outStream) const;
            std::istream &Load(std::istream &inStream, bool inTransposed = false);
            
            /*static std::ostream &SaveHeader(std::ostream &outStream, t_index expectedRows, t_index expectedCols, int version);
            static std::istream &LoadHeader(std::istream &inStream, int &outVersion, t_index &outX, t_index &outY);
            std::istream &LoadBatch(std::istream &inStream, bool inTransposed, int inVersion, t_index x, t_index y, const std::string &inCacheFileName);
*/

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

            void RandNormal(float inMean, float inStdDev, unsigned long long inSeed = 0);//normal randomess, mean 0.0f standard deviation 1.0f

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

            bool isCached(void)
            {
                return m_CacheFileHandle != -1;

                /*

                if(m_CacheFileName != "")
                {
                    t_index fsize = inDataSize*sizeof(float);

                    struct stat st;
                    fstat(m_FileCache, &st);

                    //simple check - if file is of the correct size (in the beginning file is greater by getCachedTestSize()

                    return fsize == st.st_size;

                    //int result = lseek(m_FileCache, fsize, SEEK_SET);
                    //if (result == -1)
                    //{
                    //    close(m_FileCache);
                    //    throw std::runtime_error("Error calling lseek() to 'stretch' the file");
                    //}
                    //char testData[getCachedTestSize()];

                    //ssize_t size = read(m_FileCache, testData, getCachedTestSize());
                    //
                    //if(size == getCachedTestSize())
                    //{
                    //    bool ok = true;
                    //    for(int i = 0; i < getCachedTestSize(); ++i)
                    //    {
                    //        ok = ok && testData[i] == '+';
                    //    }

                    //    return ok;
                    //}
                }

                return false;*/
            }
            /*void setCached(t_index inDataSize, bool yes)
            {
                if(m_CacheFileName != "")
                {
                    t_index size = inDataSize*sizeof(float);

                    struct stat st;
                    fstat(m_FileCache, &st);

                    if(st.st_size == size + getCachedTestSize())
                    {
                        int result = ftruncate(m_FileCache, size);
                        if (result == -1)
                        {
                            close(m_FileCache);
                            throw std::runtime_error("Error calling lseek() to 'stretch' the file");
                        }
                    }

                    fstat(m_FileCache, &st);

                    assert(st.st_size == size);

                    //ssize_t result = lseek(m_FileCache, size, SEEK_SET);
                    //if (result == -1)
                    //{
                    //    close(m_FileCache);
                    //    throw std::runtime_error("Error calling lseek() to 'stretch' the file");
                    //}

                    //for(int i = 0; i < getCachedTestSize(); ++i)
                    //{
                    //    result = write(m_FileCache, (yes) ? "+" : "-", 1);
                    //    if (result == -1)
                    //    {
                    //        close(m_FileCache);
                    //        throw std::runtime_error("Error calling write() to the file");
                    //    }
                    //}

                }
            }
            int getCachedTestSize(void)
            {
                return 4;
            }*/

            void AllocateMemory(t_index inDataSize, int inCacheFileHandle)
            {
                assert(m_CacheFileHandle == -1);

                //memory mapping
                if(inCacheFileHandle != -1)
                {
                    m_CacheFileHandle = inCacheFileHandle;

                    t_index size = inDataSize*sizeof(float);

                    m_Data = (float*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_CacheFileHandle, 0);

                    madvise(m_Data, size, MADV_RANDOM);

                    if (m_Data == MAP_FAILED)
                    {
                        close(m_CacheFileHandle);
                        throw std::runtime_error("Error mmapping the file");
                    }
                }
                else
                {
                    m_Data = new float [inDataSize];
                }

                /*if(m_CacheFileName != "")
                {
                    t_index size = inDataSize*sizeof(float);
                    //std::cout << "step 1" << std::endl;

                    //m_FileCache = open(m_CacheFileName.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
                    m_FileCache = open(m_CacheFileName.c_str(), O_RDWR | O_CREAT, (mode_t)0600);
                    if (m_FileCache == -1)
                    {
                        throw std::runtime_error("Error opening file for writing");
                    }

                    if(!isCached(inDataSize))
                    {

                        //std::cout << "NOT CACHED" << std::endl;
                        close(m_FileCache);
                        m_FileCache = open(m_CacheFileName.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
                        if (m_FileCache == -1)
                        {
                            throw std::runtime_error("Error opening file for writing");
                        }
                        setCached(inDataSize, false);
                    }
                    else
                    {
                        //std::cout << "CACHED" << std::endl;
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
                }*/
            }

            void DeallocateMemory(void)
            {
                if(m_CacheFileHandle != -1)
                {
                    int result = munmap(m_Data, m_X*m_Y*sizeof(float));
                    if (result == -1)
                    {
                        perror("Error while unmapping the data file!");
                    }
                    close(m_CacheFileHandle);
                }
                else
                {
                    delete [] m_Data;
                }

                m_Data = NULL;
            }

            void Reset(t_index inX, t_index inY, const float * inInit = NULL, int inCacheFileHandle = -1)
            {
                if(m_X*m_Y != inX*inY || m_CacheFileHandle != inCacheFileHandle)
                {
                    if(m_Data != NULL)
                    {
                        DeallocateMemory();
                    }

                    AllocateMemory(inX*inY, inCacheFileHandle);

                    assert(m_Data != NULL);
                }

                m_X = inX;
                m_Y = inY;

                if(inInit != NULL)
                {
                    memcpy(m_Data, inInit, inX*inY*sizeof(float));
                }
            }

            MatrixCpu &operator=(float inVal);
            MatrixCpu &operator=(const MatrixGpu &inMatrix);
            MatrixCpu &operator=(const MatrixCpu &inMatrix);

            MatrixCpu &operator*=(const MatrixCpu &inB);//elementwise multiplication!

            friend MatrixCpu Mult(const MatrixCpu &inA, const MatrixCpu &inB, bool transposedA = false, bool transposedB = false);//matrix multiplication!
            friend class MatrixSaverStream;
            friend class MatrixLoaderStream;

        protected:

            //static std::ostream &SaveHeader(std::ostream &outStream, t_index expectedRows, t_index expectedCols, int version = 2);
            //static void SaveHeader(const std::string &inFile, std::ofstream &outOfstream, t_index expectedRows, t_index expectedCols, int version = 3);//creates header and returns stream to write in

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
            int m_CacheFileHandle;
    };

}
void msgC(const char * inMsg, const YAMATH::MatrixCpu &x);




#endif //MATRIX_H

