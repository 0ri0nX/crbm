#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include "../myAssert.h"
#include <string>

#include "crbmComputer.h"

#include "../crbmCpu.h"

#include <exception>

using namespace std;
using namespace YAMATH;


CRBMStack::CRBMStack(int inLength, const char** inRBMFiles, int inDeviceID)
{
    try
    {
        for(int i = 0; i < inLength; ++i)
        {
            CRBM::TLayer *l = new CRBM::TLayer();
            m_Layers.push_back(l);
            l->Load(inRBMFiles[i]);
        }
    }
    catch(std::exception const& e)
    {
        //std::cerr << "Exception caught: " << e.what() << std::endl;
        Clear();
        throw e;
    }
}

CRBMStack::CRBMStack(const std::vector<string> &inRBMFiles, int inDeviceID)
{
    try
    {
        for(unsigned int i = 0; i < inRBMFiles.size(); ++i)
        {
            CRBM::TLayer *l = new CRBM::TLayer();
            m_Layers.push_back(l);
            l->Load(inRBMFiles[i]);
        }
    }
    catch(std::exception const& e)
    {
        //std::cerr << "Exception caught: " << e.what() << std::endl;
        Clear();
        throw e;
    }
}

void CRBMStack::Clear(void)
{
    for(unsigned int i = 0; i < m_Layers.size(); ++i)
    {
        delete m_Layers[i];
        m_Layers[i] = NULL;
    }

    m_Layers.clear();
}

CRBMStack::~CRBMStack(void)
{
    Clear();
}

void CRBMStack::Transform(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    ASSERT(m_Layers.size() > 0);
    ASSERT(inLenInData == GetInputSize());
    ASSERT(inLenOutData == GetOutputSize());

    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xx(1, inLenInData, inData);
    MatrixCpu y;

    for(unsigned int i = 0; i < m_Layers.size(); ++i)
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
    ASSERT(m_Layers.size() > 0 && inLenInData / GetInputSize() == inLenOutData / GetOutputSize());
    ASSERT(inLenInData % GetInputSize() == 0);
    ASSERT(inLenOutData % GetOutputSize() == 0);

    int sx = inLenInData / m_Layers[0]->getInputSize();
    int sy = m_Layers[0]->getInputSize();

    MatrixCpu xx(sx, sy, inData);
    MatrixCpu y;

    //std::cout << "c++: " << sx << "x" << sy << std::endl;

    for(unsigned int i = 0; i < m_Layers.size(); ++i)
    {
        Timer t;
        m_Layers[i]->Transform(xx, y);
        xx = y;
        cout << "Layer " << i << ": ";
        t.tac("");
    }

    ASSERT(t_index(inLenOutData) == sx*xx.getY());

    for(int i = 0; i < inLenOutData; ++i)
    {
        outData[i] = xx.getDataConst()[i];
    }
}

void CRBMStack::Transform(const std::vector<float> &inData, std::vector<float> &outData) const
{
    ASSERT(m_Layers.size() > 0);
    ASSERT(inData.size() == (unsigned int)(GetInputSize()));
    ASSERT(outData.size() == (unsigned int)(GetOutputSize()));

    //MatrixCpu line(int inX = 1, int inY = 1, const float * inInit = NULL) //column first order
    MatrixCpu xx(1, inData.size(), &inData[0]);
    MatrixCpu y;

    for(unsigned int i = 0; i < m_Layers.size(); ++i)
    {
        //cout << "   Transforming with layer " << i+1 << endl;
        m_Layers[i]->Transform(xx, y);
        xx = y;
    }

    outData.assign(xx.getDataConst(), xx.getDataConst() + xx.getY());
}

void CRBMStack::Reconstruct(int inLenInData, const float* inData, int inLenOutData, float* outData) const
{
    ASSERT(m_Layers.size() > 0);
    ASSERT(inLenInData == inLenOutData);//input is equal to output
    ASSERT(inLenOutData == GetInputSize());

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
    ASSERT(m_Layers.size() > 0);
    ASSERT(inLenInData == inLenOutData);
    ASSERT(inLenInData % GetInputSize() == 0);

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
    ASSERT(m_Layers.size() > 0);
    return  m_Layers.back()->getOutputSize();
}

int CRBMStack::GetInputSize(void) const
{
    if(m_Layers.size() > 0);
    return  m_Layers[0]->getInputSize();
}


