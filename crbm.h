#ifndef CRBM_H
#define CRBM_H

#include "matrix.h"
#include "utils.h"
#include <stdexcept>
#include <sstream>
#include "setting.h"

namespace CRBM
{

    struct CRBMLayerSetting : SettingBase
    {
        CRBMLayerSetting()
        {
            x = 200;
            y = 200;
            z = 3;
    
            cx = 10;
            cy = 10;
    
            stridex = 5;
            stridey = 5;
    
            hidden = 15;
    
            batchSize = 100;
            batchIterations = 100;
            iterations = 1000;
    
            learningRate = 0.001f;
            decayRate = 0.0f;
    
            logModulo = 50;

            saveInterval = 10;

            activationFunctionH = 0;
            activationFunctionV = 0;
    
            //momentum = 0.9f;
            //dataLimit = 0;
            //binarySampling = 0;
            //testBatchModulo = 1;
            //AFUp = AFSigmoid;
            //AFDown = AFSigmoid;
            //computeOnly = 0;
            //saveBestModel = 0;
            //testFile = "";
            //l2 = 0.0001;
            //dataType = 0; //0 - sparse: id qid:qid 1:0.12 ... , //1 - images: id qid:qid 'path-to-image'
            //imgType = 0; //0 color, 1 grey, 2 edge
        }
    
        virtual void loadFromStream(ifstream &f)
        {
            x               = loadOption(f, "x",                            x);
            y               = loadOption(f, "y",                            y);
            z               = loadOption(f, "z",                            z);
            cx              = loadOption(f, "cx",                           cx);
            cy              = loadOption(f, "cy",                           cy);
            stridex         = loadOption(f, "stridex",                      stridex);
            stridey         = loadOption(f, "stridey",                      stridey);
            hidden          = loadOption(f, "hidden",                       hidden);
            batchSize       = loadOption(f, "batchSize",                    batchSize);
            batchIterations = loadOption(f, "batchIterations",              batchIterations);
            iterations      = loadOption(f, "iterations",                   iterations);
            learningRate    = loadOption(f, "learningRate",                 learningRate);
            decayRate       = loadOption(f, "decayRate",                    decayRate);
            logModulo       = loadOption(f, "logModulo",                    logModulo);
            saveInterval    = loadOption(f, "saveInterval",                 saveInterval);
            activationFunctionH               = loadOption(f, "activationFunctionH",                            activationFunctionH);
            activationFunctionV               = loadOption(f, "activationFunctionV",                            activationFunctionV);
        }
    
        //image-size
        int x;
        int y;
        int z;
    
        //convolution-size
        int cx;
        int cy;
    
        //stride-size
        int stridex;
        int stridey;
    
        int hidden;
    
        int batchSize;
        int batchIterations;
        int iterations;
        float learningRate;
        float decayRate;

        int activationFunctionH;
        int activationFunctionV;
   
        int logModulo;
        int saveInterval;
    };


    class CRBMLayer
    {
        public:
    
            CRBMLayer(void) : m_SignalStop(false) {}
            CRBMLayer(const CRBMLayerSetting &inSetting);

            //weights are reseted only when topology changes or forced by forceResetWeights flag
            void ResetSetting(const CRBMLayerSetting &inSetting, bool forceResetWeights = false);
    
            float LearnAll(const YAMATH::MatrixGpu &inData, const std::string &inBackupFileName = "");
            float LearnBatch(const YAMATH::MatrixGpu &inBatch);
            void Transform(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData) const;
            void Reconstruct(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData);
    
            //x (width), y (height), z (depth or layer count), cx, cy is width and height of convolution filters, stridex/y are shifts of neighbour filters in x and y
            //it is expected that matrix has m.x==x and m.y == y*z
            void Convolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
            void DeConvolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch);
            void DeConvolveRaw(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
            void SetDeConvolveNormalizer(int numImages);
            void getConvolutionPatchesNumber(int &outX, int &outY) const;
    
            //all parameters are from this layer
            void RawOutput2UpperLayer(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
            //all parameters are from this layer as well
            void UpperLayer2RawOutput(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
    
            void ActivationFunction(YAMATH::MatrixGpu &inData, int inFunctionType) const;
    
            void Save(std::ostream &outStream) const;
            void Load(std::istream &inStream);
            void Save(const std::string &inName) const;
            void Load(const std::string &inName);
    
            void SignalStop(void) const;
            void ClearStop(void) const;
            bool IsStopRequired(void) const;

            void ResetWeights(void);
    
        protected:
    
            //returns setting - for convenience
            const CRBMLayerSetting& s(void) const;
    
            CRBMLayerSetting m_Setting;
    
            YAMATH::MatrixGpu m_Weights;
            YAMATH::MatrixGpu m_Normalizer;
    
            mutable bool m_SignalStop;
    };

    CRBMLayer::CRBMLayer(const CRBMLayerSetting &inSetting) :
        m_Setting(inSetting)
        , m_SignalStop(false)
    {
        ResetWeights();
    }

    void CRBMLayer::ResetSetting(const CRBMLayerSetting &inSetting, bool forceResetWeights)
    {
        bool reset = forceResetWeights
            || s().x != inSetting.x
            || s().y != inSetting.y
            || s().z != inSetting.z
            || s().cx != inSetting.cx
            || s().cy != inSetting.cy
            || s().stridex != inSetting.stridex
            || s().stridey != inSetting.stridey
            || s().hidden != inSetting.hidden;

        m_Setting = inSetting;

        if(reset)
        {
            ResetWeights();
        }
    }

    void CRBMLayer::ResetWeights(void)
    {
        m_Weights.Reset(s().cx*s().cy*s().z, s().hidden);
        m_Weights.RandNormal(0.0f, 1.0f/(10.0*s().hidden));
        cout << "weight matrix randomized!" << endl;
    }

    const CRBMLayerSetting& CRBMLayer::s(void) const
    {
        return m_Setting;
    }
    
    void CRBMLayer::SignalStop(void) const
    {
        m_SignalStop = true;
    }

    void CRBMLayer::ClearStop(void) const
    {
        m_SignalStop = false;
    }

    bool CRBMLayer::IsStopRequired(void) const
    {
        return m_SignalStop;
    }

    //a,b,c - coordinates, im - image index, x,y,z - size of image, totim - total number of images
    inline int pixelInColMajor(int a, int b, int c, int im, int x, int y, int z, int totim)
    {
        int idx = im + c*totim + a*z*totim + b*x*z*totim;
        //cout << "idx: " << idx << endl;
        return idx;
    }
    
    void CRBMLayer::getConvolutionPatchesNumber(int &outX, int &outY) const
    {
        outX = (s().x-s().cx)/s().stridex+1;
        outY = (s().y-s().cy)/s().stridey+1;
    }

    //x (width), y (height), z (depth or layer count), cx, cy is width and height of convolution filters, stridex/y are shifts of neighbour filters in x and y
    //it is expected that matrix has m.x==num.of.images and m.y == x*y*z
    void CRBMLayer::Convolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const
    {
        assert(inBatch.getY() == s().x*s().y*s().z);

        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        int numImages = inBatch.getX();
        int numPatches = nh*nv;
        int totImages = numPatches*numImages;

        outBatch.Reset(totImages , s().cx*s().cy*s().z);

//#define STREAMS_ON

#ifdef STREAMS_ON
        // allocate and initialize an array of stream handles
        int nstreams = 14;
        cout << "async " << nstreams << endl;
        cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
        for(int i = 0; i < nstreams; i++)
        {
            cudaStreamCreate(&(streams[i]));
        }
        int indexForStream = 0;
#endif //STREAMS_ON

        //TODO: remove, only for tesst
        //outBatch = -1.0;

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
#ifdef STREAMS_ON
                            cudaMemcpyAsync
#else //STREAMS_ON
                            cudaMemcpy
#endif //STREAMS_ON

                                (outBatch.getData()  + pixelInColMajor(ax, ay, az, px*numImages + py*nh*numImages, s().cx, s().cy, s().z, totImages) //convolution window target
                                     , inBatch.getDataConst() + pixelInColMajor(s().stridex*px + ax, s().stridey*py + ay, az, 0, s().x, s().y, s().z, numImages) //convolution window source
                                     , sizeof(float)*numImages
                                     , cudaMemcpyDeviceToDevice
#ifdef STREAMS_ON
                                     , streams[(++indexForStream) % nstreams]
#endif //STREAMS_ON

                                     );
                            //goto breakit;
                        }
                    }
                }
            }
        }
//breakit:

#ifdef STREAMS_ON
        // release resources
        for(int i = 0; i < nstreams; i++)
        {
            cudaDeviceSynchronize();
            cudaStreamDestroy(streams[i]);
        }
#endif //STREAMS_ON
    }

    void CRBMLayer::SetDeConvolveNormalizer(int numImages)
    {
        //is already propetly set
        if(m_Normalizer.getX() == numImages && m_Normalizer.getY() == s().x*s().y*s().z)
        {
            return;
        }

        m_Normalizer.Reset(numImages , s().x*s().y*s().z);
        m_Normalizer = 0.0f;

        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        static int ThreadsPerBlock = 512;
        int blocks = (numImages - 1) / ThreadsPerBlock + 1;

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
                            //float *dFrom = getDataConst() + pixelInColMajor(ax, ay, az, px*numImages + py*nh*numImages, cx, cy, z, totImages); //convolution window target
                            float *dTo = m_Normalizer.getData()  + pixelInColMajor(s().stridex*px + ax, s().stridey*py + ay, az, 0, s().x, s().y, s().z, numImages); //convolution window source
                            YAMATH::applyFunction<<<blocks, ThreadsPerBlock>>>(dTo, dTo, numImages, YAMATH::EFE_PlusScalar, 1.0f);
                        }
                    }
                }
            }
        }

        int num = numImages*m_Normalizer.getY();
        blocks = (num - 1) / ThreadsPerBlock + 1;
        YAMATH::applyFunction<<<blocks, ThreadsPerBlock>>>(m_Normalizer.getData(), m_Normalizer.getDataConst(), num, YAMATH::EFE_InverseAndMultiply, 1.0f);
    }

    void CRBMLayer::DeConvolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch)
    {
        //msgG("inBatch:", inBatch);
        DeConvolveRaw(inBatch, outBatch);
        //msgG("outBatch (nonnormalized):", outBatch);

        SetDeConvolveNormalizer(outBatch.getX());
        //msgG("normalizer:", m_Normalizer);

        outBatch = outBatch*m_Normalizer;
        //msgG("outBatch (normalized):", outBatch);
    }

    void CRBMLayer::DeConvolveRaw(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const
    {
        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        int numImages = inBatch.getX() / (nh*nv);

        assert(inBatch.getY() == s().cx*s().cy*s().z);

        outBatch.Reset(numImages , s().x*s().y*s().z);

        int numPatches = nh*nv;

        int totImages = numPatches*numImages;

        //initial reset to zero
        outBatch = 0.0;

        static int ThreadsPerBlock = 512;
        int blocks = (numImages - 1) / ThreadsPerBlock + 1;

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
                            YAMATH::parallelMatrixOperationBinary<<<blocks, ThreadsPerBlock>>>(dTo, dFrom, numImages, YAMATH::EFEB_Plus, dTo);
                        }
                    }
                }
            }
        }
    }

    //all parameters are from this layer
    void CRBMLayer::RawOutput2UpperLayer(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const
    {
        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        int numImages = (inBatch.getX()*inBatch.getY()) / (nh*nv*s().hidden);
        //msgG("inBatch: ", inBatch);
        //cout << "nh" << nh << endl;
        //cout << "nv" << nv << endl;
        //cout << "s().hidden" << s().hidden << endl;
        //cout << "Num images" << numImages << endl;

        int numPatches = nh*nv;
        int total = inBatch.getX()*inBatch.getY();
        int imageAllInOneSize = total/numImages;

        int features = imageAllInOneSize/numPatches;

        outBatch.Reset(numImages, imageAllInOneSize);

//#define STREAMS_ON

#ifdef STREAMS_ON
        // allocate and initialize an array of stream handles
        int nstreams = 14;
        cout << "async " << nstreams << endl;
        cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
        for(int i = 0; i < nstreams; i++)
        {
            cudaStreamCreate(&(streams[i]));
        }
        int indexForStream = 0;
#endif //STREAMS_ON

        //cout << "patches:" << numPatches << endl;
        //cout << "features:" << features << endl;
        //cout << "images:" << numImages << endl;

        for(int p = 0; p < numPatches; ++p)//p - patch number
        {
            for(int f = 0; f < features; ++f)//f - number of features (hidden layer)
            {
                        {
#ifdef STREAMS_ON
                            cudaMemcpyAsync
#else //STREAMS_ON
                            cudaMemcpy
#endif //STREAMS_ON

                                (outBatch.getData() + (f + p*features)*numImages //target
                                     , inBatch.getDataConst() + (f*numPatches + p)*numImages //source
                                     , sizeof(float)*numImages
                                     , cudaMemcpyDeviceToDevice
#ifdef STREAMS_ON
                                     , streams[(++indexForStream) % nstreams]
#endif //STREAMS_ON

                                     );
                            //goto breakit;
                        }
            }
        }
//breakit:

#ifdef STREAMS_ON
        // release resources
        for(int i = 0; i < nstreams; i++)
        {
            cudaDeviceSynchronize();
            cudaStreamDestroy(streams[i]);
        }
#endif //STREAMS_ON
    }

    //all parameters are from this layer as well
    void CRBMLayer::UpperLayer2RawOutput(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const
    {
        //horizontal and vertical number of patches
        int nh, nv;
        getConvolutionPatchesNumber(nh, nv);

        int numImages = (inBatch.getX()*inBatch.getY()) / (nh*nv*s().hidden);

        int numPatches = nh*nv;
        int total = inBatch.getX()*inBatch.getY();

        //res must be patches-number*rest ?
        outBatch.Reset(numPatches*numImages, s().hidden);

//#define STREAMS_ON

#ifdef STREAMS_ON
        // allocate and initialize an array of stream handles
        int nstreams = 14;
        cout << "async " << nstreams << endl;
        cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
        for(int i = 0; i < nstreams; i++)
        {
            cudaStreamCreate(&(streams[i]));
        }
        int indexForStream = 0;
#endif //STREAMS_ON

        //TODO: remove
        //outBatch = -1.0;

        //cout << "patches:" << numPatches << endl;
        //cout << "features:" << s().hidden << endl;
        //cout << "images:" << numImages << endl;

        for(int p = 0; p < numPatches; ++p)//p - patch number
        {
            for(int f = 0; f < s().hidden; ++f)//f - number of features (hidden layer)
            {
                        {
#ifdef STREAMS_ON
                            cudaMemcpyAsync
#else //STREAMS_ON
                            cudaMemcpy
#endif //STREAMS_ON

                                (outBatch.getData() + (f*numPatches + p)*numImages //target
                                     , inBatch.getDataConst() + (f + p*s().hidden)*numImages //source
                                     , sizeof(float)*numImages
                                     , cudaMemcpyDeviceToDevice
#ifdef STREAMS_ON
                                     , streams[(++indexForStream) % nstreams]
#endif //STREAMS_ON

                                     );
                            //goto breakit;
                        }
            }
        }
//breakit:

#ifdef STREAMS_ON
        // release resources
        for(int i = 0; i < nstreams; i++)
        {
            cudaDeviceSynchronize();
            cudaStreamDestroy(streams[i]);
        }
#endif //STREAMS_ON

    }

    float computeError(const YAMATH::MatrixGpu &inInp, const YAMATH::MatrixGpu &inOut)
    {
        YAMATH::MatrixGpu r2, r3;
        r2 = inInp - inOut;
        r2 ^= 2.0f;
        r3 = r2.Sum();
    
        YAMATH::MatrixCpu rr = r3;
    
        return rr.getDataConst()[0]/inInp.getX();
    }

    float computeWeightSize(const YAMATH::MatrixGpu &inW)
    {
        YAMATH::MatrixGpu r2, r3;
        r2 = inW;
        r2 ^= 2.0f;
        r3 = r2.Sum();
    
        YAMATH::MatrixCpu rr = r3;
    
        return rr.getDataConst()[0]/(inW.getX()*inW.getY());
    }

    float CRBMLayer::LearnAll(const YAMATH::MatrixGpu &inData, const std::string &inBackupFileName)
    {
        int transX, transY;//transformed size
        getConvolutionPatchesNumber(transX, transY);

        cout << "On image " << s().x << "x" << s().y << "x" << s().z << " applied convolution " << s().cx << "x" << s().cy << " with stride " << s().stridex << "x" << s().stridey
             << " => " << transX << "x" << transY << " patches." << endl;

        float error = -1;
        for(int i = 1; i <= s().iterations && !IsStopRequired(); ++i)
        {
            Timer t;
            cout << i << " / " << s().iterations << " sampling ... " << flush;
            YAMATH::MatrixGpu batch = inData.Sample(s().batchSize);
            t.tac();

            error = LearnBatch(batch);

            if(i % s().saveInterval == 0 && inBackupFileName != "")
            {
                Save(inBackupFileName);
            }
        }

        return error;
    }

    float CRBMLayer::LearnBatch(const YAMATH::MatrixGpu &inBatch)
    {
        Timer timer;

        YAMATH::MatrixGpu x, xraw, y, x2, y2, dw1, dw2;

        //timer.tic();
        cout << "    Preparing data ... " << flush;
        Convolve(inBatch, x);
        timer.tac();

        float weightSize = computeWeightSize(m_Weights);
        cout << "    Weights' size: [" << weightSize << "]" << endl;

        //msgG("x", x);
        //msgG("w", m_Weights);

        timer.tic();

        float error = -1.0f;
        cout << "    " << s().batchIterations << " iterations:" << flush;

        for(int i = 1; i <= s().batchIterations && !IsStopRequired(); ++i)
        {
            y = Mult(x, m_Weights);
            ActivationFunction(y, s().activationFunctionH);

            x2 = Mult(y, m_Weights.T());
            ActivationFunction(x2, s().activationFunctionV);

            y2 = Mult(x2, m_Weights);
            ActivationFunction(y2, s().activationFunctionH);

            dw1 = Mult(x.T(), y);
            dw2 = Mult(x2.T(), y2);

            dw1 *= (s().learningRate/x.getX());
            dw2 *= (s().learningRate/x.getX());

            if(s().decayRate > 0.0f)
            {
                m_Weights *= (1.0f - s().decayRate);
            }

            m_Weights += dw1;
            m_Weights -= dw2;

            if(i % s().logModulo == 0 || i == s().batchIterations || i == 1)
            {
                error = computeError(x, x2);
                if(i != 1)
                {
                    cout << ",";
                }

                cout << " (" << i << ") " << error << flush;
            }
        }
        timer.tac("     ... in ");

        return error;
    }

    void CRBMLayer::ActivationFunction(YAMATH::MatrixGpu &inData, int inFunctionType) const
    {
        switch(inFunctionType)
        {
            case 0: //linear
                break;
            case 1: //sigmoid
                inData = inData.Sigmoid();
                break;
            case 2: //rectified linear
                inData = inData.Minimally(0.0f);
                break;
            default:
                assert(0);// && "unknown activation function ID");
        }
    }

    void  CRBMLayer::Transform(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData) const
    {
        YAMATH::MatrixGpu x, y;

        Convolve(inData, x);

        y = Mult(x, m_Weights);
        ActivationFunction(y, s().activationFunctionH);

        RawOutput2UpperLayer(y, outData);
    }

    void CRBMLayer::Reconstruct(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData)
    {
        YAMATH::MatrixGpu x, y;

        //msgG("inData", inData);

        UpperLayer2RawOutput(inData, y);

        //msgG("y", inData);

        x = Mult(y, m_Weights.T());
        ActivationFunction(x, s().activationFunctionV);

        //msgG("x", x);

        DeConvolve(x, outData);
    }

    template<typename T>
    void sv(std::ostream &out, const std::string &inName, const T &inValue)
    {
        out << inName;
        out << " " << inValue << std::endl;
    }
   
    template<>
    void sv<>(std::ostream &out, const std::string &inName, const YAMATH::MatrixGpu &inValue)
    {
        out << inName << " ";
        YAMATH::MatrixCpu m = inValue;
        m.Save(out);
        out << endl;
    }

    template<typename T>
    void checkVal(const T &wanted, const T &got, const std::string &name = "")
    {
        if(wanted != got)
        {
            std::stringstream e;
            if(name != "")
            {
                e << "in [" << name << "]";
            }
            e << "wanted [" << wanted << "] but got [" << got << "]" << std::endl;

            throw runtime_error(e.str());
        }
    }

    template<typename T>
    void checkValRange(const T &wantedMin, const T &wantedMax, const T &got, const std::string &name = "")
    {
        if(got < wantedMin || got > wantedMax)
        {
            std::stringstream e;
            if(name != "")
            {
                e << "in [" << name << "]";
            }
            e << "wanted [" << wantedMin << " .. " << wantedMax << "] but got [" << got << "]" << std::endl;

            throw runtime_error(e.str());
        }
    }

    template<typename T>
    void lvc(std::istream &in, const std::string &inName, const T &inMinValue, const T &inMaxValue, T &outValue)
    {
        std::string name;
        in >> name >> outValue;
        checkVal(inName, name);
        checkValRange(inMinValue, inMaxValue, outValue, inName);
    }

    template<typename T>
    void lv(std::istream &in, const std::string &inName, T &outValue)
    {
        std::string name;
        in >> name >> outValue;
        
        checkVal(inName, name);
    }
   
    template<>
    void lv<>(std::istream &in, const std::string &inName, YAMATH::MatrixGpu &outValue)
    {
        std::string name;
        in >> name;
        assert(name == inName);

        YAMATH::MatrixCpu m;
        m.Load(in);

        outValue = m;
    }

    void CRBMLayer::Save(std::ostream &out) const
    {
        sv(out, "CRBMLayer", 3);

        sv(out, "learningSpeed", s().learningRate);

        sv(out, "x", s().x);
        sv(out, "y", s().y);
        sv(out, "z", s().z);

        sv(out, "cx", s().cx);
        sv(out, "cy", s().cy);

        sv(out, "stridex", s().stridex);
        sv(out, "stridey", s().stridey);

        sv(out, "hidden", s().hidden);

        sv(out, "weights", m_Weights);

        sv(out, "activationFunctionH", s().activationFunctionH);
        sv(out, "activationFunctionV", s().activationFunctionV);

        sv(out, "decayRate", s().decayRate);
    }

    void CRBMLayer::Load(std::istream &in)
    {
        int version = -1;
        lvc(in, "CRBMLayer", 1, 3, version);

        lv(in, "learningSpeed", m_Setting.learningRate);

        lv(in, "x", m_Setting.x);
        lv(in, "y", m_Setting.y);
        lv(in, "z", m_Setting.z);

        lv(in, "cx", m_Setting.cx);
        lv(in, "cy", m_Setting.cy);

        lv(in, "stridex", m_Setting.stridex);
        lv(in, "stridey", m_Setting.stridey);

        lv(in, "hidden", m_Setting.hidden);

        lv(in, "weights", m_Weights);
        
        if(version > 1)
        {
            lv(in, "activationFunctionH", m_Setting.activationFunctionH);
            lv(in, "activationFunctionV", m_Setting.activationFunctionV);
        }

        if(version > 2)
        {
            lv(in, "decayRate", m_Setting.decayRate);
        }
    }

    void CRBMLayer::Save(const std::string &inName) const
    {
        std::cout << "saving [" << inName << "] ... " << std::flush;
        Timer t;

        ofstream f(inName.c_str());
        Save(f);
        f.close();

        t.tac();
    }

    void CRBMLayer::Load(const std::string &inName)
    {
        std::cout << "Loading RBM layer [" << inName << "] ... " << std::flush;
        Timer t;

        ifstream f(inName.c_str());
        Load(f);
        f.close();

        t.tac();
    }
}//namespace CRBM

#endif //CRBM_H
