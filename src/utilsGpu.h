#ifndef UTILSGPU_H
#define UTILSGPU_H

#include "matrix.h"
#include "utils.h"
#include <time.h>
#include <string>
#include <iostream>

void msgG(const char * inMsg, const YAMATH::MatrixGpu &inM)
{
    YAMATH::MatrixCpu x = inM;
    msgC(inMsg, x);
}

void saveMatrix(const YAMATH::MatrixGpu &inM, const std::string &filename)
{
    YAMATH::MatrixCpu resx = inM;
    saveMatrix(resx, filename);
}

//load from stream (gpu matrix)
template<>
void lv<>(std::istream &in, const std::string &inName, YAMATH::MatrixGpu &outValue)
{
    std::string name;
    in >> name;
    in.ignore(1);
    assert(name == inName);

    YAMATH::MatrixCpu m;
    m.Load(in);

    outValue = m;
}

//save to stream (gpu matrix)
template<>
void sv<>(std::ostream &out, const std::string &inName, const YAMATH::MatrixGpu &inValue)
{
    out << inName << " ";
    YAMATH::MatrixCpu m = inValue;
    m.Save(out);
    out << std::endl;
}


#endif //UTILSGPU_H
