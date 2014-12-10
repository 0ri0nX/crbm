#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>

#include "rbmComputer.h"

#include "matrix.h"
using namespace std;
using namespace YAMATH;


//std::vector<GPUMatrix*> m_Weights;
//typedef MatrixGpu GPUMatrix;

void loadMatrix(MatrixCpu &inM, const string& filename, bool inTransposed = false)
{
    //cout << "loading [" << filename << "] ... " << endl;
    ifstream f(filename.c_str());
    inM.Load(f, inTransposed);
    f.close();
    //cout << " ... " << inM.getX() << "x" << inM.getY() << endl;
}

RBMStack::RBMStack(int inLength, const char** inWeights, int inGpuID)
{
    using namespace YAMATH;

    cudaSetDevice(inGpuID);

    //cublasStatus_t stat;
    //cublasHandle_t handle;

    //stat = cublasCreate(&handle);
    //if (stat != CUBLAS_STATUS_SUCCESS) {
    //    printf ("CUBLAS initialization failed\n");
    //    return EXIT_FAILURE;
    //}

    MatrixCpu xCpu;

    for(int i = 0; i < inLength; ++i)
    {
        loadMatrix(xCpu, inWeights[i]);
        //cout << i << ". ";
        //msgC("weights", *xCpu);
        MatrixGpu *w = new MatrixGpu(xCpu);
        m_Weights.push_back(w);
    }
}

RBMStack::RBMStack(const std::vector<string> &inWeights, int inGpuID)
{
    using namespace YAMATH;

    cudaSetDevice(inGpuID);

    //cublasStatus_t stat;
    //cublasHandle_t handle;

    //stat = cublasCreate(&handle);
    //if (stat != CUBLAS_STATUS_SUCCESS) {
    //    printf ("CUBLAS initialization failed\n");
    //    return EXIT_FAILURE;
    //}

    MatrixCpu xCpu;

    for(int i = 0; i < inWeights.size(); ++i)
    {
        loadMatrix(xCpu, inWeights[i]);
        //cout << i << ". ";
        //msgC("weights", *xCpu);
        MatrixGpu *w = new MatrixGpu(xCpu);
        m_Weights.push_back(w);
    }
}

RBMStack::~RBMStack(void)
{
    for(int i = 0; i < m_Weights.size(); ++i)
    {
        delete m_Weights[i];
        m_Weights[i] = NULL;
    }
}

void RBMStack::Transform(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xCpu(1, inLenInData, inData);
    MatrixGpu xx = xCpu;
    MatrixGpu y;

    for(int i = 0; i < m_Weights.size(); ++i)
    {
        //cout << "Transforming with weights " << i+1 << endl;
        //cout << xx.getX() << "x" << xx.getY() << " -- " << m_Weights[i]->getX() << "x" << m_Weights[i]->getY() << endl;
        y = Mult(xx, *(m_Weights[i]));
        xx = y;
    }

    MatrixCpu resx = xx;

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = resx.getDataConst()[i];
    }
}

void RBMStack::Transform(const std::vector<float> &inData, std::vector<float> &outData) const
{
    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xx(1, inData.size(), &inData[0]);
    MatrixGpu y;

    for(int i = 0; i < m_Weights.size(); ++i)
    {
        //cout << "Transforming with weights " << i+1 << endl;
        y = Mult(xx, *(m_Weights[i]));
        xx = y;
    }

    MatrixCpu resx = xx;
    outData.assign(resx.getDataConst(), resx.getDataConst() + resx.getY());
}

int RBMStack::GetOutputSize(void) const
{
    return m_Weights.back()->getY();
}


//void RBMStack::Reconstruct(const std::vector<float> &inData, std::vector<float> &outData)
//{
//    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
//    MatrixCpu xx(1, iinData.size(), &inData[0])
//    MatrixGpu y;
//
//    for(int i = m_Weights.size() - 1; i >= 0; --i)
//    {
//        cout << "Reconstructing with weights " << i+1 << endl;
//        y = Mult(xx, (weights[i])->T());
//        xx = y;
//    }
//
//    MatrixCpu resx = xx;
//    outData.assign(resx.getDataConst(), resx.getDataConst() + resx.getY());
//}

