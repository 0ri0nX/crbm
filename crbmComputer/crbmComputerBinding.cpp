#include "crbmComputer.h"

extern "C"
{
    CRBMStack* CRBMStack_new(int inLength, const char** inWeights, int inDeviceID)
    {
        return new CRBMStack(inLength, inWeights, inDeviceID);
    }

    int CRBMStack_GetOutputSize(CRBMStack* inCRBMStack)
    {
        return inCRBMStack->GetOutputSize();
    }

    int CRBMStack_GetInputSize(CRBMStack* inCRBMStack)
    {
        return inCRBMStack->GetInputSize();
    }

    void CRBMStack_Transform(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        inCRBMStack->Transform(inLenInData, inData, inLenOutData, outData);
    }

    void CRBMStack_TransformBatch(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        inCRBMStack->TransformBatch(inLenInData, inData, inLenOutData, outData);
    }

    void CRBMStack_Reconstruct(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        inCRBMStack->Reconstruct(inLenInData, inData, inLenOutData, outData);
    }

    void CRBMStack_ReconstructBatch(CRBMStack *inCRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        inCRBMStack->ReconstructBatch(inLenInData, inData, inLenOutData, outData);
    }

    void CRBMStack_delete(CRBMStack* inCRBMStack)
    {
        delete inCRBMStack;
    }
}

