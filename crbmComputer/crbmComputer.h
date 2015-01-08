#ifndef CRBM_COMPUTER_H
#define CRBM_COMPUTER_H

#include <vector>
#include <string>


namespace CRBM
{
#ifdef CUDA
    class CRBMLayerGpu;
    typedef CRBMLayerGpu TLayer;
#else
    class CRBMLayerCpu;
    typedef CRBMLayerCpu TLayer;
#endif
}

class CRBMStack
{
    private:
        //std::vector<CRBM::CRBMLayerGpu*> m_Layers;
        std::vector<CRBM::TLayer*> m_Layers;

    public:
        CRBMStack(const std::vector<std::string> &inRBMFiles, int inDeviceID = 0);
        CRBMStack(int inLength, const char** inRBMFiles, int inDeviceID = 0);

        int GetOutputSize(void) const;
        int GetInputSize(void) const;
        ~CRBMStack(void);

        void Transform(const std::vector<float> &inData, std::vector<float> &outData) const;
        void Transform(int inLenInData, const float* inData, int inLenOutData, float* outData) const;
        void TransformBatch(int inLenInData, const float* inData, int inLenOutData, float* outData) const;
        void Reconstruct(int inLenInData, const float* inData, int inLenOutData, float* outData) const;
        void ReconstructBatch(int inLenInData, const float* inData, int inLenOutData, float* outData) const;

};

#endif //CRBM_COMPUTER_H

