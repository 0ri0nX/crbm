#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>

#include "crbmComputer.h"

#include "../src/crbm.h"
using namespace std;
using namespace YAMATH;


CRBMStack::CRBMStack(int inLength, const char** inRBMFiles, int inGpuID)
{
    cudaSetDevice(inGpuID);

    for(int i = 0; i < inLength; ++i)
    {
        CRBM::CRBMLayer *l = new CRBM::CRBMLayer();
        l->Load(inRBMFiles[i]);
        m_Layers.push_back(l);
    }
}

CRBMStack::CRBMStack(const std::vector<string> &inRBMFiles, int inGpuID)
{
    cudaSetDevice(inGpuID);

    for(int i = 0; i < inRBMFiles.size(); ++i)
    {
        CRBM::CRBMLayer *l = new CRBM::CRBMLayer();
        l->Load(inRBMFiles[i]);
        m_Layers.push_back(l);
    }
}

CRBMStack::~CRBMStack(void)
{
    for(int i = 0; i < m_Layers.size(); ++i)
    {
        delete m_Layers[i];
        m_Layers[i] = NULL;
    }
}

void CRBMStack::Transform(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    assert(inLenInData == m_Layers[0]->getInputSize());

    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xCpu(1, inLenInData, inData);
    MatrixGpu xx = xCpu;
    MatrixGpu y;

    for(int i = 0; i < m_Layers.size(); ++i)
    {
        Timer t;
        //cout << "   Transforming with layer " << i+1 << endl;
        m_Layers[i]->Transform(xx, y);
        xx = y;
        cout << "Layer " << i << ": ";
        t.tac("");
    }

    MatrixCpu resx = xx;

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = resx.getDataConst()[i];
    }
}

void CRBMStack::TransformBatch(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    assert(inLenInData % m_Layers[0]->getInputSize() == 0);

    int sx = inLenInData / m_Layers[0]->getInputSize();
    int sy = m_Layers[0]->getInputSize();

    MatrixCpu xCpu(sx, sy, inData);
    MatrixGpu xx = xCpu;
    MatrixGpu y;

    std::cout << "c++: " << sx << "x" << sy << std::endl;

    for(int i = 0; i < m_Layers.size(); ++i)
    {
        Timer t;
        m_Layers[i]->Transform(xx, y);
        xx = y;
        cout << "Layer " << i << ": ";
        t.tac("");
    }

    MatrixCpu resx = xx;

    assert(inLenOutData == sx*resx.getY());

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = resx.getDataConst()[i];
    }
}

void CRBMStack::Transform(const std::vector<float> &inData, std::vector<float> &outData) const
{
    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xCpu(1, inData.size(), &inData[0]);
    MatrixGpu xx = xCpu;
    MatrixGpu y;

    for(int i = 0; i < m_Layers.size(); ++i)
    {
        //cout << "   Transforming with layer " << i+1 << endl;
        m_Layers[i]->Transform(xx, y);
        xx = y;
    }

    MatrixCpu resx = xx;
    outData.assign(resx.getDataConst(), resx.getDataConst() + resx.getY());
}

void CRBMStack::Reconstruct(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xCpu(1, inLenInData, inData);
    MatrixGpu xx = xCpu;
    MatrixGpu y;

    for(int i = m_Layers.size() - 1; i >=0; ++i)
    {
        Timer t;
        m_Layers[i]->Reconstruct(xx, y);
        xx = y;
        cout << "Layer " << i << ": ";
        t.tac("");
    }

    MatrixCpu resx = xx;

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = resx.getDataConst()[i];
    }
}

int CRBMStack::GetOutputSize(void) const
{
    return  m_Layers.back()->getOutputSize();
}

int CRBMStack::GetInputSize(void) const
{
    return  m_Layers[0]->getInputSize();
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

