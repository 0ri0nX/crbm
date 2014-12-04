#ifndef CRBM_H
#define CRBM_H

#include "matrix.h"

namespace CRBM
{

class CRBMLayer
{
    public:

        CRBMLayer(int x, int y, int z, int cx, int cy, int stridex, int stridey, int hidden);

        float learnAll(const YAMATH::MatrixGpu &inData, int batchSize = 256, int globalIterations = 1000, int batchIterations = 100);
        float learnBatch(const YAMATH::MatrixGpu &inBatch, int batchIterations = 100);
        void transform(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData) const;
        void reconstruct(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData);

        //x (width), y (height), z (depth or layer count), cx, cy is width and height of convolution filters, stridex/y are shifts of neighbour filters in x and y
        //it is expected that matrix has m.x==x and m.y == y*z
        void Convolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
        void DeConvolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch);
        void DeConvolveRaw(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
        void SetDeConvolveNormalizer(int numImages);
        void ConvolutionPatchesNumber(int &outX, int &outY) const;

        //all parameters are from this layer
        void RawOutput2UpperLayer(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
        //all parameters are from this layer as well
        void UpperLayer2RawOutput(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;

        static void FunctionHidden(YAMATH::MatrixGpu &inHidden);
        static void FunctionVisible(YAMATH::MatrixGpu &inVisible);

    protected:

        float m_LearningSpeed; // = 0.001f

        //image-size
        int m_x;// = 200;
        int m_y;// = 200;
        int m_z;// = 3;

        //convolution-size
        int m_cx;// = 10;
        int m_cy;// = 10;

        //stride-size
        int m_stridex;// = 5;
        int m_stridey;// = 5;

        int m_hidden;// = 15

        YAMATH::MatrixGpu m_Weights;
        YAMATH::MatrixGpu m_Normalizer;
};

    CRBMLayer::CRBMLayer(int x, int y, int z, int cx, int cy, int stridex, int stridey, int hidden)
        : m_x(x), m_y(y), m_z(z), m_cx(cx), m_cy(cy), m_stridex(stridex), m_stridey(stridey), m_Weights(cx*cy*z, hidden), m_Normalizer(1)
    {
        m_Weights.RandNormal(0.0f, 1.0f/(10.0*hidden));
        cout << "weight matrix randomized!" << endl;
    }
    
    //a,b,c - coordinates, im - image index, x,y,z - size of image, totim - total number of images
    inline int pixelInColMajor(int a, int b, int c, int im, int x, int y, int z, int totim)
    {
        int idx = im + c*totim + a*z*totim + b*x*z*totim;
        //cout << "idx: " << idx << endl;
        return idx;
    }
    
    void CRBMLayer::ConvolutionPatchesNumber(int &outX, int &outY) const
    {
        outX = (m_x-m_cx)/m_stridex+1;
        outY = (m_y-m_cy)/m_stridey+1;
    }

    //x (width), y (height), z (depth or layer count), cx, cy is width and height of convolution filters, stridex/y are shifts of neighbour filters in x and y
    //it is expected that matrix has m.x==num.of.images and m.y == x*y*z
    void CRBMLayer::Convolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const
    {
        assert(inBatch.getY() == m_x*m_y*m_z);

        //horizontal and vertical number of patches
        int nh, nv;
        ConvolutionPatchesNumber(nh, nv);

        int numImages = inBatch.getX();
        int numPatches = nh*nv;
        int totImages = numPatches*numImages;

        outBatch.Reset(totImages , m_cx*m_cy*m_z);

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
        outBatch = -1.0;

        for(int py = 0; py < nv; ++py)//y - order of convolution window - patch y
        {
            for(int px = 0; px < nh; ++px)//x - order of convolution window - patch x
            {
                for(int ay = 0; ay < m_cy; ++ay)//y in convolution window
                {
                    for(int ax = 0; ax < m_cx; ++ax)//x in convolution window
                    {
                        for(int az = 0; az < m_z; ++az)//image layers
                        {
#ifdef STREAMS_ON
                            cudaMemcpyAsync
#else //STREAMS_ON
                            cudaMemcpy
#endif //STREAMS_ON

                                (outBatch.getData()  + pixelInColMajor(ax, ay, az, px*numImages + py*nh*numImages, m_cx, m_cy, m_z, totImages) //convolution window target
                                     , inBatch.getDataConst() + pixelInColMajor(m_stridex*px + ax, m_stridey*py + ay, az, 0, m_x, m_y, m_z, numImages) //convolution window source
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
        if(m_Normalizer.getX() == numImages && m_Normalizer.getY() == m_x*m_y*m_z)
        {
            return;
        }

        m_Normalizer.Reset(numImages , m_x*m_y*m_z);

        //horizontal and vertical number of patches
        int nh, nv;
        ConvolutionPatchesNumber(nh, nv);

        static int ThreadsPerBlock = 512;
        int blocks = (numImages - 1) / ThreadsPerBlock + 1;

        for(int py = 0; py < nv; ++py)//y - order of convolution window - patch y
        {
            for(int px = 0; px < nh; ++px)//x - order of convolution window - patch x
            {
                for(int ay = 0; ay < m_cy; ++ay)//y in convolution window
                {
                    for(int ax = 0; ax < m_cx; ++ax)//x in convolution window
                    {
                        for(int az = 0; az < m_z; ++az)//image layers
                        {
                            //float *dFrom = getDataConst() + pixelInColMajor(ax, ay, az, px*numImages + py*nh*numImages, cx, cy, z, totImages); //convolution window target
                            float *dTo = m_Normalizer.getData()  + pixelInColMajor(m_stridex*px + ax, m_stridey*py + ay, az, 0, m_x, m_y, m_z, numImages); //convolution window source
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
        DeConvolveRaw(inBatch, outBatch);

        cout << outBatch.getX() << " x " << outBatch.getY() << endl;
        cout << m_Normalizer.getX() << " x " << m_Normalizer.getY() << endl;

        outBatch = outBatch*m_Normalizer;
    }

    void CRBMLayer::DeConvolveRaw(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const
    {
        //horizontal and vertical number of patches
        int nh, nv;
        ConvolutionPatchesNumber(nh, nv);

        int numImages = inBatch.getX() / (nh*nv);

        assert(inBatch.getY() == m_cx*m_cy*m_z);

        outBatch.Reset(numImages , m_x*m_y*m_z);

        int numPatches = nh*nv;

        int totImages = numPatches*numImages;

        //TODO: remove
        outBatch = 0.0;

        static int ThreadsPerBlock = 512;
        int blocks = (numImages - 1) / ThreadsPerBlock + 1;

        for(int py = 0; py < nv; ++py)//y - order of convolution window - patch y
        {
            for(int px = 0; px < nh; ++px)//x - order of convolution window - patch x
            {
                for(int ay = 0; ay < m_cy; ++ay)//y in convolution window
                {
                    for(int ax = 0; ax < m_cx; ++ax)//x in convolution window
                    {
                        for(int az = 0; az < m_z; ++az)//image layers
                        {
                            float *dFrom = inBatch.getDataConst() + pixelInColMajor(ax, ay, az, px*numImages + py*nh*numImages, m_cx, m_cy, m_z, totImages); //convolution window target
                            float *dTo = outBatch.getData()  + pixelInColMajor(m_stridex*px + ax, m_stridey*py + ay, az, 0, m_x, m_y, m_z, numImages); //convolution window source
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
        ConvolutionPatchesNumber(nh, nv);

        int numImages = (inBatch.getX()*inBatch.getY()) / (nh*nv*m_hidden);

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

        //TODO: remove
        outBatch = -1.0;

        cout << "patches:" << numPatches << endl;
        cout << "features:" << features << endl;
        cout << "images:" << numImages << endl;

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
        ConvolutionPatchesNumber(nh, nv);

        int numImages = (inBatch.getX()*inBatch.getY()) / (nh*nv*m_hidden);

        int numPatches = nh*nv;
        int total = inBatch.getX()*inBatch.getY();

        //res must be patches-number*rest ?
        outBatch.Reset(numPatches*numImages, m_hidden);

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
        outBatch = -1.0;

        cout << "patches:" << numPatches << endl;
        cout << "features:" << m_hidden << endl;
        cout << "images:" << numImages << endl;

        for(int p = 0; p < numPatches; ++p)//p - patch number
        {
            for(int f = 0; f < m_hidden; ++f)//f - number of features (hidden layer)
            {
                        {
#ifdef STREAMS_ON
                            cudaMemcpyAsync
#else //STREAMS_ON
                            cudaMemcpy
#endif //STREAMS_ON

                                (outBatch.getData() + (f*numPatches + p)*numImages //target
                                     , inBatch.getDataConst() + (f + p*m_hidden)*numImages //source
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

    float CRBMLayer::learnBatch(const YAMATH::MatrixGpu &inBatch, int batchIterations)
    {
        int LOG_MODULO = 10;
        //Timer timer;

        int transX, transY;//transformed size
        ConvolutionPatchesNumber(transX, transY);

        cout << "On image " << m_x << "x" << m_y << "x" << m_z << " applied convolution " << m_cx << "x" << m_cy << " with stride " << m_stridex << "x" << m_stridey << endl;
        cout << "It resulted into " << transX << "x" << transY << " patches." << endl;

        YAMATH::MatrixGpu x, xraw, y, x2, y2, dw1, dw2, err, lastW;

        //timer.tic();
        Convolve(inBatch, x);
        //timer.tac("Convolve: ");

        //lastW = m_Weights;

        bool ONE_ROW = true;
        float error = -1.0f;

        for(int i = 0; i < batchIterations; ++i)
        {
            y = Mult(x, m_Weights);
            FunctionHidden(y);

            x2 = Mult(y, m_Weights.T());
            FunctionVisible(x2);

            y2 = Mult(x2, m_Weights);
            FunctionHidden(y2);

            dw1 = Mult(x.T(), y);
            dw2 = Mult(x2.T(), y2);

            dw1 *= (m_LearningSpeed/x.getX());
            dw2 *= (m_LearningSpeed/x.getX());

            m_Weights = m_Weights + dw1;
            m_Weights = m_Weights - dw2;

            //lastW *= 0.00001;
            //w = w - lastW;
            //lastW = w;

            if(i % LOG_MODULO == 0 || i+1 == batchIterations)
            {
                error = computeError(x, x2);
                cout << i << ": " << error << flush;

                if(ONE_ROW)
                {
                    cout << "                  " << "\r" << flush;
                }
                else
                {
                    cout << endl;
                }
            }
        }
        cout << endl;

        return error;
    }

    void CRBMLayer::FunctionHidden(YAMATH::MatrixGpu &inHidden)
    {
        //inHidden = inHidden.Sigmoid();
    }
    void CRBMLayer::FunctionVisible(YAMATH::MatrixGpu &inVisible)
    {
        //inVisible = inVisible.Sigmoid();
    }

    void  CRBMLayer::transform(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData) const
    {
        YAMATH::MatrixGpu x, y;

        Convolve(inData, x);

        y = Mult(x, m_Weights);
        FunctionHidden(y);

        RawOutput2UpperLayer(y, outData);
    }

    void CRBMLayer::reconstruct(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData)
    {
        YAMATH::MatrixGpu x, y;

        UpperLayer2RawOutput(inData, y);

        x = Mult(y, m_Weights.T());
        FunctionVisible(x);

        DeConvolve(x, outData);
    }



}//namespace CRBM

#endif //CRBM_H
