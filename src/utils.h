#ifndef UTILS_H
#define UTILS_H

#include "matrixCpu.h"
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


void loadMatrix(YAMATH::MatrixCpu &inM, const std::string& filename, bool inTransposed = false, const std::string &inCacheFileName = "")
{
    std::cout << "loading [" << filename << "] ... " << std::flush;
    Timer t;
    std::ifstream f(filename.c_str());
    inM.Load(f, inTransposed, inCacheFileName);
    f.close();
    std::cout << inM.getX() << " x " << inM.getY() << "  ";
    t.tac();
    //msgC(filename.c_str(), inM);
}

void saveMatrix(const YAMATH::MatrixCpu &inM, const std::string &filename)
{
    std::cout << "saving [" << filename << "] ... " << std::flush;
    Timer t;
    std::ofstream f(filename.c_str());
    inM.Save(f);
    f.close();
    t.tac();
    //msgC(filename.c_str(), inM);
}

#endif //UTILS_H
