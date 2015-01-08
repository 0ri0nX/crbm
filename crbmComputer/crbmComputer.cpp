#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>

#include "crbmComputer.h"

#include "../src/crbmCpu.h"

using namespace std;
using namespace YAMATH;


CRBMStack::CRBMStack(int inLength, const char** inRBMFiles, int inDeviceID)
{
    for(int i = 0; i < inLength; ++i)
    {
        CRBM::TLayer *l = new CRBM::TLayer();
        l->Load(inRBMFiles[i]);
        m_Layers.push_back(l);
    }
}

CRBMStack::CRBMStack(const std::vector<string> &inRBMFiles, int inDeviceID)
{
    for(int i = 0; i < inRBMFiles.size(); ++i)
    {
        CRBM::TLayer *l = new CRBM::TLayer();
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
    MatrixCpu xx(1, inLenInData, inData);
    MatrixCpu y;

    for(int i = 0; i < m_Layers.size(); ++i)
    {
        Timer t;
        //cout << "   Transforming with layer " << i+1 << endl;
        m_Layers[i]->Transform(xx, y);
        xx = y;
        cout << "Layer " << i << ": ";
        t.tac("");
    }

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = xx.getDataConst()[i];
    }
}

void CRBMStack::TransformBatch(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    assert(inLenInData % m_Layers[0]->getInputSize() == 0);

    int sx = inLenInData / m_Layers[0]->getInputSize();
    int sy = m_Layers[0]->getInputSize();

    MatrixCpu xx(sx, sy, inData);
    MatrixCpu y;

    //std::cout << "c++: " << sx << "x" << sy << std::endl;

    for(int i = 0; i < m_Layers.size(); ++i)
    {
        Timer t;
        m_Layers[i]->Transform(xx, y);
        xx = y;
        cout << "Layer " << i << ": ";
        t.tac("");
    }

    assert(inLenOutData == sx*xx.getY());

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = xx.getDataConst()[i];
    }
}

void CRBMStack::Transform(const std::vector<float> &inData, std::vector<float> &outData) const
{
    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xx(1, inData.size(), &inData[0]);
    MatrixCpu y;

    for(int i = 0; i < m_Layers.size(); ++i)
    {
        //cout << "   Transforming with layer " << i+1 << endl;
        m_Layers[i]->Transform(xx, y);
        xx = y;
    }

    outData.assign(xx.getDataConst(), xx.getDataConst() + xx.getY());
}

void CRBMStack::Reconstruct(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xx(1, inLenInData, inData);
    MatrixCpu y;

    for(int i = m_Layers.size() - 1; i >=0; --i)
    {
        Timer t;
        m_Layers[i]->Reconstruct(xx, y);
        xx = y;
        cout << "Layer " << i << ": ";
        t.tac("");
    }

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = xx.getDataConst()[i];
    }
}

void CRBMStack::ReconstructBatch(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    assert(inLenInData % m_Layers.back()->getOutputSize() == 0);

    int sx = inLenInData / m_Layers.back()->getOutputSize();
    int sy = m_Layers.back()->getOutputSize();

    MatrixCpu xx(sx, sy, inData);
    MatrixCpu y;

    for(int i = m_Layers.size() - 1; i >=0; --i)
    {
        Timer t;
        m_Layers[i]->Reconstruct(xx, y);
        xx = y;
        cout << "Layer " << i << ": ";
        t.tac("");
    }

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = xx.getDataConst()[i];
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


