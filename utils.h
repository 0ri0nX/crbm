#ifndef UTILS_H
#define UTILS_H

#include "matrix.h"
#include <time.h>
#include <string>
#include <iostream>

class Timer
{
    public:
        Timer(void)
        {
            tic();
        }

        void tic(void)
        {
            clock_gettime(CLOCK_MONOTONIC, &m_TimeSpec);
        }

        void tac(const std::string &inComment = "")
        {
            timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);

            float t = (ts.tv_sec - m_TimeSpec.tv_sec) + (ts.tv_nsec - m_TimeSpec.tv_nsec)/10e9;

            std::cout << inComment << t << " sec" << std::endl;
        }

    private:

        timespec m_TimeSpec;
};


using namespace YAMATH;

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
        std::cout  << inMsg << ":[" << x.getDataConst()[0] << "]" << flush;
    }
    else
    {
        std::cout  << inMsg << ":" << std::endl;
        x.Save(std::cout);
        std::cout << std::endl;
    }
}

void msgG(const char * inMsg, const YAMATH::MatrixGpu &inM)
{
    YAMATH::MatrixCpu x = inM;
    msgC(inMsg, x);
}

void loadMatrix(YAMATH::MatrixCpu &inM, const std::string& filename, bool inTransposed = false)
{
    std::cout << "loading [" << filename << "] ... " << std::endl;
    Timer t;
    ifstream f(filename.c_str());
    inM.Load(f, inTransposed);
    f.close();
    t.tac("   ... done in ");
    msgC(filename.c_str(), inM);
}

void saveMatrix(YAMATH::MatrixCpu &inM, const std::string &filename)
{
    std::cout << "saving [" << filename << "] ... " << std::endl;
    Timer t;
    ofstream f(filename.c_str());
    inM.Save(f);
    f.close();
    t.tac("   ... done in ");
    msgC(filename.c_str(), inM);
}

void saveGpuMatrix(YAMATH::MatrixGpu &inM, const std::string &filename)
{
    YAMATH::MatrixCpu resx = inM;
    saveMatrix(resx, filename);
}

#endif //UTILS_H
