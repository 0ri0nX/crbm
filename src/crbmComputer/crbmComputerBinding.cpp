#include "crbmComputerBinding.h"
#include "crbmComputer.h"
#include <exception>
#include <iostream>

extern "C"
{
    CRBMStack* CRBMStack_new(int inLength, const char** inWeights, int inDeviceID)
    {
        CRBMStack * p = NULL;
        try
        {
            p = new CRBMStack(inLength, inWeights, inDeviceID);
        }
        catch(std::exception const& e)
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            p = NULL;
        }

        return p;
    }

    int CRBMStack_GetOutputSize(CRBMStack* inCRBMStack)
    {
        int res = 0;
        try
        {
            res = inCRBMStack->GetOutputSize();
        }
        catch(std::exception const& e)
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            res = 0;
        }

        return res;
    }

    int CRBMStack_GetInputSize(CRBMStack* inCRBMStack)
    {
        int res = 0;
        try
        {
            res = inCRBMStack->GetInputSize();
        }
        catch(std::exception const& e)
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            res = 0;
        }

        return res;
    }

    void CRBMStack_Transform(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        try
        {
            inCRBMStack->Transform(inLenInData, inData, inLenOutData, outData);
        }
        catch(std::exception const& e)
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
        }
    }

    void CRBMStack_TransformBatch(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        try
        {
            inCRBMStack->TransformBatch(inLenInData, inData, inLenOutData, outData);
        }
        catch(std::exception const& e)
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
        }
    }

    void CRBMStack_Reconstruct(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        try
        {
            inCRBMStack->Reconstruct(inLenInData, inData, inLenOutData, outData);
        }
        catch(std::exception const& e)
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
        }
    }

    void CRBMStack_ReconstructBatch(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        try
        {
            inCRBMStack->ReconstructBatch(inLenInData, inData, inLenOutData, outData);
        }
        catch(std::exception const& e)
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
        }
    }

    void CRBMStack_delete(CRBMStack* inCRBMStack)
    {
        if(inCRBMStack != NULL)
        {
            delete inCRBMStack;
        }
    }
}

