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
        void transform(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData);
        float reconstruct(const YAMATH::MatrixGpu &inData, YAMATH::MatrixGpu &outData);



        //x (width), y (height), z (depth or layer count), cx, cy is width and height of convolution filters, stridex/y are shifts of neighbour filters in x and y
        //it is expected that matrix has m.x==x and m.y == y*z
        void Convolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
        void DeConvolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch);
        void DeConvolveRaw(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const;
        void SetDeConvolveNormalizer(int numImages);

        //all parameters are from this layer
        void RawOutput2UpperLayer(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch, int numImages) const;
        //all parameters are from this layer as well
        void UpperLayer2RawOutput(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch, int numImages) const;

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
    : m_x(x), m_y(y), m_z(z), m_cx(cx), m_cy(cy), m_stridex(stridex), m_stridey(stridey), m_Weights(1), m_Normalizer(1)
{
}


//a,b,c - coordinates, im - image index, x,y,z - size of image, totim - total number of images
inline int pixelInColMajor(int a, int b, int c, int im, int x, int y, int z, int totim)
{
    int idx = im + c*totim + a*z*totim + b*x*z*totim;
    //cout << "idx: " << idx << endl;
    return idx;
}

void convolutionPatchesNumber(int x, int y, int z, int cx, int cy, int stridex, int stridey, int &outX, int &outY)
{

    outX = (x-cx)/stridex+1;
    outY = (y-cy)/stridey+1;
}

    //x (width), y (height), z (depth or layer count), cx, cy is width and height of convolution filters, stridex/y are shifts of neighbour filters in x and y
    //it is expected that matrix has m.x==num.of.images and m.y == x*y*z
    void CRBMLayer::Convolve(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch) const
    {
        assert(inBatch.getY() == m_x*m_y*m_z);

        //horizontal and vertical number of patches
        int nh = (m_x-m_cx)/m_stridex+1;
        int nv = (m_y-m_cy)/m_stridey+1;

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
        int nh = (m_x-m_cx)/m_stridex+1;
        int nv = (m_y-m_cy)/m_stridey+1;

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
        int nh = (m_x-m_cx)/m_stridex+1;
        int nv = (m_y-m_cy)/m_stridey+1;

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
    void CRBMLayer::RawOutput2UpperLayer(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch, int numImages) const
    {
        //horizontal and vertical number of patches
        int nh = (m_x-m_cx)/m_stridex+1;
        int nv = (m_y-m_cy)/m_stridey+1;

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
    void CRBMLayer::UpperLayer2RawOutput(const YAMATH::MatrixGpu &inBatch, YAMATH::MatrixGpu &outBatch, int numImages) const
    {
        //horizontal and vertical number of patches
        int nh = (m_x-m_cx)/m_stridex+1;
        int nv = (m_y-m_cy)/m_stridey+1;

        int numPatches = nh*nv;
        int total = inBatch.getX()*inBatch.getY();
        int imageAllInOneSize = total/numImages;
        //int totImages = numPatches*numImages;

        int features = imageAllInOneSize/numPatches;

        //res must be patches-number*rest
        outBatch.Reset(numPatches*numImages, features);

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

                                (outBatch.getData() + (f*numPatches + p)*numImages //target
                                     , inBatch.getDataConst() + (f + p*features)*numImages //source
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
}//namespace CRBM

#endif //CRBM_H
