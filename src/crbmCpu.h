#ifndef CRBMCPU_H
#define CRBMCPU_H

#include "crbm.h"

namespace CRBM
{
    class CRBMLayerCpu : public CRBMLayer
    {
        public:
    
            CRBMLayerCpu(void) : CRBMLayer() {}
            CRBMLayerCpu(const CRBMLayerSetting &inSetting);

            void Transform(const YAMATH::MatrixCpu &inData, YAMATH::MatrixCpu &outData) const;
            void Reconstruct(const YAMATH::MatrixCpu &inData, YAMATH::MatrixCpu &outData);
    
            //x (width), y (height), z (depth or layer count), cx, cy is width and height of convolution filters, stridex/y are shifts of neighbour filters in x and y
            //it is expected that matrix has m.x==x and m.y == y*z
            void Convolve(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch) const;
            void DeConvolve(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch);
            void DeConvolveRaw(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch) const;
            void SetDeConvolveNormalizer(int numImages);

            //all parameters are from this layer
            void RawOutput2UpperLayer(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch) const;
            //all parameters are from this layer as well
            void UpperLayer2RawOutput(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch) const;
    
            void ActivationFunction(YAMATH::MatrixCpu &inData, int inFunctionType) const;
    
            virtual void SaveSpecific(std::ostream &outStream) const;
            virtual void LoadSpecific(std::istream &inStream);

            virtual void ResetWeights(void);
    
         protected:
            YAMATH::MatrixCpu m_Weights;
            YAMATH::MatrixCpu m_Normalizer;
    };

    CRBMLayerCpu::CRBMLayerCpu(const CRBMLayerSetting &inSetting) :
        CRBMLayer(inSetting)
    {
        ResetWeights();
    }

    void CRBMLayerCpu::ResetWeights(void)
    {
        m_Weights.Reset(s().cx*s().cy*s().z, s().hidden);

        m_Weights.RandNormal(0.0f, 1.0f/(10.0*s().hidden));

        std::cout << "weight matrix randomized!" << std::endl;
    }

    //x (width), y (height), z (depth or layer count), cx, cy is width and height of convolution filters, stridex/y are shifts of neighbour filters in x and y
    //it is expected that matrix has m.x==num.of.images and m.y == x*y*z
    void CRBMLayerCpu::Convolve(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch) const
    {
        assert(inBatch.getY() == YAMATH::t_index(s().x*s().y*s().z));

        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        int numImages = inBatch.getX();
        int numPatches = nh*nv;
        int totImages = numPatches*numImages;

        outBatch.Reset(totImages , s().cx*s().cy*s().z);

        //99 3 4 1 2 5     one extreme
        for(int ay = 0; ay < s().cy; ++ay)//y in convolution window //3
        {
            for(int ax = 0; ax < s().cx; ++ax)//x in convolution window //4
            {
                for(int py = 0; py < nv; ++py)//y - order of convolution window - patch y //1
                {
                    for(int px = 0; px < nh; ++px)//x - order of convolution window - patch x //2
                    {
                        for(int az = 0; az < s().z; ++az)//image layers //5
                        {
        //45629 5 4 2 1 3  second extreme
        //for(int az = 0; az < s().z; ++az)//image layers //5
        //{
        //    for(int ax = 0; ax < s().cx; ++ax)//x in convolution window //4
        //    {
        //        for(int px = 0; px < nh; ++px)//x - order of convolution window - patch x //2
        //        {
        //            for(int py = 0; py < nv; ++py)//y - order of convolution window - patch y //1
        //            {
        //                for(int ay = 0; ay < s().cy; ++ay)//y in convolution window //3
        //                {
                            //std::cout << pixelInColMajor(s().stridex*px + ax, s().stridey*py + ay, az, 0, s().x, s().y, s().z, numImages);
                            //std::cout << " " << py;
                            //std::cout << " " << px;
                            //std::cout << " " << ay;
                            //std::cout << " " << ax;
                            //std::cout << " " << az;
                            //std::cout << std::endl;
                            memcpy
                                (outBatch.getData()  + pixelInColMajor(ax, ay, az, px*numImages + py*nh*numImages, s().cx, s().cy, s().z, totImages) //convolution window target
                                     , inBatch.getDataConst() + pixelInColMajor(s().stridex*px + ax, s().stridey*py + ay, az, 0, s().x, s().y, s().z, numImages) //convolution window source
                                     , sizeof(float)*numImages
                                     );
                        }
                    }
                }
            }
        }
    }

    void CRBMLayerCpu::SetDeConvolveNormalizer(int numImages)
    {
        //is already propetly set
        if(m_Normalizer.getX() == YAMATH::t_index(numImages) && m_Normalizer.getY() == YAMATH::t_index(s().x*s().y*s().z))
        {
            return;
        }
        //cout << " numImages=" << numImages << ", x=" << s().x << ", y=" << s().y << ", z=" << s().z << endl;

        m_Normalizer.Reset(numImages , s().x*s().y*s().z);
        m_Normalizer = 0.0f;

        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        for(int py = 0; py < nv; ++py)//y - order of convolution window - patch y
        {
            for(int px = 0; px < nh; ++px)//x - order of convolution window - patch x
            {
                for(int ay = 0; ay < s().cy; ++ay)//y in convolution window
                {
                    for(int ax = 0; ax < s().cx; ++ax)//x in convolution window
                    {
                        for(int az = 0; az < s().z; ++az)//image layers
                        {
                            float *dTo = m_Normalizer.getData()  + pixelInColMajor(s().stridex*px + ax, s().stridey*py + ay, az, 0, s().x, s().y, s().z, numImages); //convolution window source
                            for(int i = 0; i < numImages; ++i)
                            {
                                dTo[i] += 1.0f;
                            }
                            //YAMATH::applyFunction<<<blocks, ThreadsPerBlock>>>(dTo, dTo, numImages, YAMATH::EFE_PlusScalar, 1.0f);
                        }
                    }
                }
            }
        }

        int num = numImages*m_Normalizer.getY();

        for(int i = 0; i < num; ++i)
        {
            m_Normalizer.getData()[i] = 1.0f/m_Normalizer.getDataConst()[i];
        }
        //YAMATH::applyFunction<<<blocks, ThreadsPerBlock>>>(m_Normalizer.getData(), m_Normalizer.getDataConst(), num, YAMATH::EFE_InverseAndMultiply, 1.0f);
    }

    void CRBMLayerCpu::DeConvolve(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch)
    {
        //msgG("inBatch:", inBatch);
        DeConvolveRaw(inBatch, outBatch);
        //msgG("outBatch (nonnormalized):", outBatch);

        SetDeConvolveNormalizer(outBatch.getX());
        //msgG("normalizer:", m_Normalizer);

        outBatch *= m_Normalizer;
        //msgG("outBatch (normalized):", outBatch);
    }

    void CRBMLayerCpu::DeConvolveRaw(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch) const
    {
        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        int numImages = inBatch.getX() / (nh*nv);

        assert(inBatch.getY() == YAMATH::t_index(s().cx*s().cy*s().z));

        outBatch.Reset(numImages , s().x*s().y*s().z);

        int numPatches = nh*nv;

        int totImages = numPatches*numImages;

        //initial reset to zero
        outBatch = 0.0;

        for(int py = 0; py < nv; ++py)//y - order of convolution window - patch y
        {
            for(int px = 0; px < nh; ++px)//x - order of convolution window - patch x
            {
                for(int ay = 0; ay < s().cy; ++ay)//y in convolution window
                {
                    for(int ax = 0; ax < s().cx; ++ax)//x in convolution window
                    {
                        for(int az = 0; az < s().z; ++az)//image layers
                        {
                            float *dFrom = inBatch.getDataConst() + pixelInColMajor(ax, ay, az, px*numImages + py*nh*numImages, s().cx, s().cy, s().z, totImages); //convolution window target
                            float *dTo = outBatch.getData()  + pixelInColMajor(s().stridex*px + ax, s().stridey*py + ay, az, 0, s().x, s().y, s().z, numImages); //convolution window source
                            for(int i = 0; i < numImages; ++i)
                            {
                                dTo[i] += dFrom[i];
                            }
                            //YAMATH::parallelMatrixOperationBinary<<<blocks, ThreadsPerBlock>>>(dTo, dFrom, numImages, YAMATH::EFEB_Plus, dTo);
                        }
                    }
                }
            }
        }
    }

    //all parameters are from this layer
    void CRBMLayerCpu::RawOutput2UpperLayer(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch) const
    {
        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        int numImages = (inBatch.getX()*inBatch.getY()) / (nh*nv*s().hidden);
        //msgG("inBatch: ", inBatch);
        //cout << "nh" << nh << std::endl;
        //cout << "nv" << nv << std::endl;
        //cout << "s().hidden" << s().hidden << std::endl;
        //cout << "Num images" << numImages << std::endl;

        int numPatches = nh*nv;
        int total = inBatch.getX()*inBatch.getY();
        int imageAllInOneSize = total/numImages;

        int features = imageAllInOneSize/numPatches;

        outBatch.Reset(numImages, imageAllInOneSize);

        //cout << "patches:" << numPatches << std::endl;
        //cout << "features:" << features << std::endl;
        //cout << "images:" << numImages << std::endl;
        
        //best 1 2
        //worst 2 1

        //TT_I;
        for(int p = 0; p < numPatches; ++p)//p - patch number //1
        {
            for(int f = 0; f < features; ++f)//f - number of features (hidden layer) //2
            {
                        {
                            memcpy

                                (outBatch.getData() + (f + p*features)*numImages //target
                                     , inBatch.getDataConst() + (f*numPatches + p)*numImages //source
                                     , sizeof(float)*numImages
                                     );
                        }
            }
        }
        //TT_M("Total measured time RawOutput2UpperLayer()")
    }

    //all parameters are from this layer as well
    void CRBMLayerCpu::UpperLayer2RawOutput(const YAMATH::MatrixCpu &inBatch, YAMATH::MatrixCpu &outBatch) const
    {
        //horizontal and vertical number of patches
        int nh, nv;

        getConvolutionPatchesNumber(nh, nv);

        //std::cout << "nh=" << nh << ", nv=" << nv << ", s().hidden=" << s().hidden << ", inBatch.getX()=" << inBatch.getX() << ", inBatch.getY()=" << inBatch.getY() << std::endl;

        int numImages = (inBatch.getX()*inBatch.getY()) / (nh*nv*s().hidden);

        int numPatches = nh*nv;
        //YAMATH::t_index total = inBatch.getX()*inBatch.getY();

        //res must be patches-number*rest ?
        outBatch.Reset(numPatches*numImages, s().hidden);

        //TODO: remove
        //outBatch = -1.0;

        //cout << "patches:" << numPatches << std::endl;
        //cout << "features:" << s().hidden << std::endl;
        //cout << "images:" << numImages << std::endl;

        for(int p = 0; p < numPatches; ++p)//p - patch number
        {
            for(int f = 0; f < s().hidden; ++f)//f - number of features (hidden layer)
            {
                        {
                            memcpy
                                (outBatch.getData() + (f*numPatches + p)*numImages //target
                                     , inBatch.getDataConst() + (f + p*s().hidden)*numImages //source
                                     , sizeof(float)*numImages
                                     );
                        }
            }
        }
    }

    
//    void testNan(const YAMATH::MatrixCpu &inInp)
//    {
//        YAMATH::MatrixCpu r;
//        r = inInp.Sum();
//        YAMATH::MatrixCpu rr = r;
//
//        assert(inInp.getX()*inInp.getY() > 0);
//    
//        float res = rr.getDataConst()[0]/(inInp.getX()*inInp.getY());
//
//        if(isnan(res) || isinf(res))
//        {
//            std::cout << "Returned " << res << " when computing error!" << std::endl;
//
//            //YAMATH::MatrixCpu zz = inInp;
//            //for(YAMATH::t_index i = 0; i < zz.getX()*zz.getY(); ++i)
//            //{
//            //    cout << " " << zz.getDataConst()[i];
//            //}
//
//            assert(0);
//        }
//    }

    void CRBMLayerCpu::ActivationFunction(YAMATH::MatrixCpu &inData, int inFunctionType) const
    {
        switch(inFunctionType)
        {
            case 0: //linear
                break;
            case 1: //sigmoid
                for(YAMATH::t_index i = 0; i < inData.getX()*inData.getY(); ++i)
                {
                    inData.getData()[i] = 1.0f/(1.0f + exp(inData.getDataConst()[i]));
                }
                break;
            case 2: //rectified linear
                for(YAMATH::t_index i = 0; i < inData.getX()*inData.getY(); ++i)
                {
                    if(inData.getDataConst()[i] < 0.0f)
                    {
                        inData.getData()[i] = 0.0f;
                    }
                }
                break;
            default:
                assert(0);// && "unknown activation function ID");
        }
    }

//#define T_I Timer TTT;
//#define T_M(x) TTT.tac(x ": "); TTT.tic();

#define T_I
#define T_M(x)

    void  CRBMLayerCpu::Transform(const YAMATH::MatrixCpu &inData, YAMATH::MatrixCpu &outData) const
    {
        T_I;
        YAMATH::MatrixCpu x, y;
        T_M("\nInit");

        Convolve(inData, x);
        T_M("Conv");

        y = Mult(x, m_Weights);
        T_M("Mult");

        ActivationFunction(y, s().activationFunctionH);
        T_M("ActFunc");

        RawOutput2UpperLayer(y, outData);
        T_M("Deconv");
    }

    void CRBMLayerCpu::Reconstruct(const YAMATH::MatrixCpu &inData, YAMATH::MatrixCpu &outData)
    {
        YAMATH::MatrixCpu x, y;

        //msgG("inData", inData);

        UpperLayer2RawOutput(inData, y);

        //msgG("y", inData);

        //x = Mult(y, m_Weights.T());
        x = Mult(y, m_Weights, false, true);
        ActivationFunction(x, s().activationFunctionV);

        //msgG("x", x);

        DeConvolve(x, outData);
    }

   
    void CRBMLayerCpu::SaveSpecific(std::ostream &out) const
    {
        sv(out, "weights", m_Weights);
    }

    void CRBMLayerCpu::LoadSpecific(std::istream &in)
    {
        lv(in, "weights", m_Weights);
    }


}//namespace CRBM

#endif //CRBMCPU_H
