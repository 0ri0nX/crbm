#ifndef RBM_COMPUTER_H
#define RBM_COMPUTER_H

#include <vector>
#include <string>

namespace YAMATH
{
    class MatrixGpu;
}

class RBMStack
{
    private:
        std::vector<YAMATH::MatrixGpu*> m_Weights;

    public:
        RBMStack(const std::vector<std::string> &inWeights, int inGpuID = 0);
        RBMStack(int inLength, const char** inWeights, int inGpuID = 0);

        int GetOutputSize(void) const;
        ~RBMStack(void);

        void Transform(const std::vector<float> &inData, std::vector<float> &outData) const;
        void Transform(int inLenInData, const float* inData, int inLenOutData, float* outData) const;

};

#endif //RBM_COMPUTER_H

