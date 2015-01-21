#ifndef CRBM_H
#define CRBM_H

#include "matrixCpu.h"
#include "utils.h"
#include <stdexcept>
#include <sstream>
#include "setting.h"
#include <cmath>

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
            incrementalSave = 0;
            incrementalSaveStart = 1;

            activationFunctionH = 0;
            activationFunctionV = 0;

            momentum = 0.0f;

            noiseCenterRange = 0.0f;//computes mean and dev of data. center of noise is range (0-noiseCenterRange)*dev
            noiseDevRange = 0.0f;//scale of noise is dev*noiseDevRange
    
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
    
        virtual void loadFromStream(std::ifstream &f)
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
            incrementalSave = loadOption(f, "incrementalSave",              incrementalSave);
            incrementalSaveStart            = loadOption(f, "incrementalSaveStart",                     incrementalSaveStart);
            activationFunctionH             = loadOption(f, "activationFunctionH",                      activationFunctionH);
            activationFunctionV             = loadOption(f, "activationFunctionV",                      activationFunctionV);
            momentum        = loadOption(f, "momentum",                     momentum);
            noiseCenterRange= loadOption(f, "noiseCenterRange",             noiseCenterRange   );
            noiseDevRange   = loadOption(f, "noiseDevRange",                noiseDevRange   );
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
        int incrementalSaveStart;
        int incrementalSave;

        float momentum;
        float noiseCenterRange;
        float noiseDevRange;
    };


    class CRBMLayer
    {
        public:
    
            CRBMLayer(void) : m_SignalStop(false) {}
            CRBMLayer(const CRBMLayerSetting &inSetting);

            //weights are reseted only when topology changes or forced by forceResetWeights flag
            void ResetSetting(const CRBMLayerSetting &inSetting, bool forceResetWeights = false);
    
            void getConvolutionPatchesNumber(int &outX, int &outY) const;

            //returns size of transformed image (outx*outy*outz)
            int getOutputSize(void) const;
            //returns size of input image (inx*iny*inz)
            int getInputSize(void) const;
   
            virtual void SaveSpecific(std::ostream &outStream) const = 0;
            virtual void LoadSpecific(std::istream &inStream) = 0;

            void Save(std::ostream &outStream) const;
            void Load(std::istream &inStream);
            void Save(const std::string &inName) const;
            void Load(const std::string &inName);
    
            void SignalStop(void) const;
            void ClearStop(void) const;
            bool IsStopRequired(void) const;

            virtual void ResetWeights(void) = 0;
    
            //returns setting - for convenience
            const CRBMLayerSetting& s(void) const;

         protected:
   
            CRBMLayerSetting m_Setting;
    
            mutable bool m_SignalStop;
    };

    CRBMLayer::CRBMLayer(const CRBMLayerSetting &inSetting) :
        m_Setting(inSetting)
        , m_SignalStop(false)
    {
        //ResetWeights();
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
        //cout << "idx: " << idx << std::endl;
        return idx;
    }
    
    void CRBMLayer::getConvolutionPatchesNumber(int &outX, int &outY) const
    {
        outX = (s().x-s().cx)/s().stridex+1;
        outY = (s().y-s().cy)/s().stridey+1;
    }

    int CRBMLayer::getOutputSize(void) const
    {
        int outx, outy;

        getConvolutionPatchesNumber(outx, outy);

        return outx*outy*s().hidden;
    }

    int CRBMLayer::getInputSize(void) const
    {
        return s().x*s().y*s().z;
    }

    void CRBMLayer::Save(std::ostream &out) const
    {
        int version = 7;

        sv(out, "CRBMLayer", version);

        //1
        sv(out, "learningSpeed", s().learningRate);

        sv(out, "x", s().x);
        sv(out, "y", s().y);
        sv(out, "z", s().z);

        sv(out, "cx", s().cx);
        sv(out, "cy", s().cy);

        sv(out, "stridex", s().stridex);
        sv(out, "stridey", s().stridey);

        sv(out, "hidden", s().hidden);

        SaveSpecific(out);

        //2
        sv(out, "activationFunctionH", s().activationFunctionH);
        sv(out, "activationFunctionV", s().activationFunctionV);

        //3
        sv(out, "decayRate", s().decayRate);

        //4
        sv(out, "saveInterval", s().saveInterval);
        sv(out, "incrementalSave", s().incrementalSave);
        sv(out, "incrementalSaveStart", s().incrementalSaveStart);

        //5 matrix versioning

        //6
        sv(out, "momentum", s().momentum);

        //7
        sv(out, "noiseCenterRange", s().noiseCenterRange);
        sv(out, "noiseDevRange", s().noiseDevRange);
    }

    void CRBMLayer::Load(std::istream &in)
    {
        int version = -1;
        int minVersion = 1;
        int maxVersion = 7;
        lvc(in, "CRBMLayer", minVersion, maxVersion, version);

        lv(in, "learningSpeed", m_Setting.learningRate);

        lv(in, "x", m_Setting.x);
        lv(in, "y", m_Setting.y);
        lv(in, "z", m_Setting.z);

        lv(in, "cx", m_Setting.cx);
        lv(in, "cy", m_Setting.cy);

        lv(in, "stridex", m_Setting.stridex);
        lv(in, "stridey", m_Setting.stridey);

        lv(in, "hidden", m_Setting.hidden);

        LoadSpecific(in);
        
        if(version >= 2)
        {
            lv(in, "activationFunctionH", m_Setting.activationFunctionH);
            lv(in, "activationFunctionV", m_Setting.activationFunctionV);
        }

        if(version >= 3)
        {
            lv(in, "decayRate", m_Setting.decayRate);
        }

        if(version >= 4)
        {
            lv(in, "saveInterval", m_Setting.saveInterval);
            lv(in, "incrementalSave", m_Setting.incrementalSave);
            lv(in, "incrementalSaveStart", m_Setting.incrementalSaveStart);
        }

        //5 - matrix version

        if(version >= 6)
        {
            lv(in, "momentum", m_Setting.momentum);
        }

        if(version >= 7)
        {
            lv(in, "noiseCenterRange", m_Setting.noiseCenterRange);
            lv(in, "noiseDevRange", m_Setting.noiseDevRange);
        }
    }

    void CRBMLayer::Save(const std::string &inName) const
    {
        std::cout << "saving [" << inName << "] ... " << std::flush;
        Timer t;

        std::ofstream f(inName.c_str(), std::ios::binary);
        Save(f);
        f.close();

        t.tac();
    }

    void CRBMLayer::Load(const std::string &inName)
    {
        std::cout << "Loading RBM layer [" << inName << "] ... " << std::flush;
        Timer t;

        std::ifstream f(inName.c_str(), std::ios::binary);
        Load(f);
        f.close();

        t.tac();
    }
}//namespace CRBM

#endif //CRBM_H
