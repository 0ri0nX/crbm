#ifndef CRBM_COMPUTER_H
#define CRBM_COMPUTER_H

#include <vector>
#include <string>

namespace CRBM
{
    class CRBMLayer;
}

class CRBMStack
{
    private:
        std::vector<CRBM::CRBMLayer*> m_Layers;

    public:
        CRBMStack(const std::vector<std::string> &inRBMFiles, int inGpuID = 0);
        CRBMStack(int inLength, const char** inRBMFiles, int inGpuID = 0);

        int GetOutputSize(void) const;
        ~CRBMStack(void);

        void Transform(const std::vector<float> &inData, std::vector<float> &outData) const;
        void Transform(int inLenInData, const float* inData, int inLenOutData, float* outData) const;

};

#endif //CRBM_COMPUTER_H

