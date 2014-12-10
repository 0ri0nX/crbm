#include "rbmComputer.h"

extern "C"
{
    RBMStack* RBMStack_new(int inLength, const char** inWeights, int inGpuID)
    {
        return new RBMStack(inLength, inWeights, inGpuID);
    }

    int RBMStack_GetOutputSize(RBMStack* inRBMStack)
    {
        return inRBMStack->GetOutputSize();
    }

    void RBMStack_Transform(RBMStack *inRBMStack, int inLenInData, const float* inData, int inLenOutData, float* outData)
    {
        inRBMStack->Transform(inLenInData, inData, inLenOutData, outData);
    }

    void RBMStack_delete(RBMStack* inRBMStack)
    {
        delete inRBMStack;
    }
}

