#ifndef UTILSGPU_H
#define UTILSGPU_H

#include "matrix.h"
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

#endif //UTILSGPU_H
