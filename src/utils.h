#ifndef UTILS_H
#define UTILS_H

#include "myAssert.h"
#include "matrixCpu.h"
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>

//#define LOAD_VERBOSITY 2
#define LOAD_VERBOSITY 1

//helper class for timing
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


void loadMatrix(YAMATH::MatrixCpu &inM, const std::string& filename, bool inTransposed = false);

void saveMatrix(const YAMATH::MatrixCpu &inM, const std::string &filename, int inVersion = 3);

//save to stream
template<typename T>
void sv(std::ostream &out, const std::string &inName, const T &inValue)
{
    out << inName;
    out << " " << inValue << std::endl;
}

template<typename T>
void checkVal(const T &wanted, const T &got, const std::string &name = "")
{
    if(wanted != got)
    {
        std::stringstream e;
        if(name != "")
        {
            e << "in [" << name << "]";
        }
        e << "wanted [" << wanted << "] but got [" << got << "]" << std::endl;

        throw std::runtime_error(e.str());
    }
}

template<typename T>
void checkValRange(const T &wantedMin, const T &wantedMax, const T &got, const std::string &name = "")
{
    if(got < wantedMin || got > wantedMax)
    {
        std::stringstream e;
        if(name != "")
        {
            e << "in [" << name << "]";
        }
        e << "wanted [" << wantedMin << " .. " << wantedMax << "] but got [" << got << "]" << std::endl;

        throw std::runtime_error(e.str());
    }
}

//load from stream with check
template<typename T>
void lvc(std::istream &in, const std::string &inName, const T &inMinValue, const T &inMaxValue, T &outValue)
{
#if LOAD_VERBOSITY > 1
    std::cout << "Loading [" << inName << "] ..." << std::flush;
#endif

    std::string name;
    in >> name >> outValue;
    checkVal(inName, name);
    checkValRange(inMinValue, inMaxValue, outValue, inName);

#if LOAD_VERBOSITY > 1
    std::cout << " done" << std::endl;
#endif
}

//load from stream
template<typename T>
void lv(std::istream &in, const std::string &inName, T &outValue)
{
#if LOAD_VERBOSITY > 1
    std::cout << "Loading [" << inName << "] ..." << std::flush;
#endif

    std::string name;
    in >> name >> outValue;
    
    checkVal(inName, name);

#if LOAD_VERBOSITY > 1
    std::cout << " done" << std::endl;
#endif
}
 
//load from stream (cpu matrix)
template<>
void lv<>(std::istream &in, const std::string &inName, YAMATH::MatrixCpu &outValue);


//save to stream (cpu matrix)
template<>
void sv<>(std::ostream &out, const std::string &inName, const YAMATH::MatrixCpu &inValue);

#endif //UTILS_H
