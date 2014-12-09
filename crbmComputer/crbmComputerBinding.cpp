#include "crbmComputer.h"

extern "C"
{
    CRBMStack* RBMStack_new(int inLength, const char** inWeights, int inGpuID)
    {
        return new CRBMStack(inLength, inWeights, inGpuID);
    }

    int CRBMStack_GetOutputSize(CRBMStack* inCRBMStack)
    {
        return inCRBMStack->GetOutputSize();
    }

    void CRBMStack_Transform(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        inCRBMStack->Transform(inLenInData, inData, inLenOutData, outData);
    }

    void CRBMStack_delete(CRBMStack* inCRBMStack)
    {
        delete inCRBMStack;
    }
}

