#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>
#include <time.h>

using namespace std;

/*
//#define CUDA

#ifdef CUDA
    //#include "deviceGPU.h"
    #include<cublas_v2.h>


#else
    #include "deviceCPU.h"
    typedef ComputingDevice::DeviceCPU dev;
#endif
*/

#include "matrix.h"

#include "crbm.h"


using namespace YAMATH;

typedef MatrixGpu Mat;

float computeError(Mat &inInp, Mat &inOut)
{
    Mat r2, r3;
    r2 = inInp - inOut;
    //msgG("in", inInp);
    //msgG("out", inOut);
    //msgG("r2", r2);
    r2 ^= 2.0f;
    r3 = r2.Sum();
    r3 *= 1.0f / inInp.getX();

    MatrixCpu rr = r3;

    msgG("abssum2", r3);

    return rr.getDataConst()[0];
}

int main(int argc, char** argv)
{
/*    if(argc != 7 && argc != 8)
    {
        cout << "Too few params!" << endl;
        cout << argv[0] << " input-vector-file input-weight-file hidden-size learning-speed iter batch [cudadevice-id]" << endl;
        exit(1);
    }

    if(argc > 7)
    {
        int device = atoi(argv[7]);
        cout << "Device ID: " << device << endl;
        cudaSetDevice(device);
    }

    cublasStatus_t stat;
    cublasHandle_t handle;

    cout << "cublas init ..." << flush;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    cout << " done" << endl;

    int hidden = atoi(argv[3]);
    float lSpeed = atof(argv[4]);
    float iterations = atof(argv[5]);
    int batchSize = atoi(argv[6]);

    MatrixCpu *xCpu = new MatrixCpu();

    loadMatrix(*xCpu, argv[1]);

    int rows = xCpu->getX();
    int cols = xCpu->getY();

    Mat xx = *xCpu;
    msgG("loaded", xx);

    delete xCpu;
    xCpu = new MatrixCpu();

    if(string(argv[2]) != "-")
    {
        loadMatrix(*xCpu, string(argv[2]));
        msgC("w", *xCpu);
    }

//#define TEST
#ifdef TEST
    //image-size
    int im_x = 3;
    int im_y = 4;
    int im_z = 2;

    //convolution-size
    int im_cx = 2;
    int im_cy = 2;

    //stride-size
    int im_stridex = 1;
    int im_stridey = 1;
#else
    //image-size
    int im_x = 200;
    int im_y = 200;
    int im_z = 3;

    //convolution-size
    int im_cx = 10;
    int im_cy = 10;

    //stride-size
    int im_stridex = 5;
    int im_stridey = 5;
#endif
    CRBM::CRBMLayer abc(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey, 15);

    Timer timer;

    int transX, transY;

    convolutionPatchesNumber(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey, transX, transY);

    cout << "On image " << im_x << "x" << im_y << "x" << im_z << " applied convolution " << im_cx << "x" << im_cy << " with stride " << im_stridex << "x" << im_stridey << endl;
    cout << "It resulted into " << transX << "x" << transY << " patches." << endl;

    //int pn = transX*transY;

    Mat x, xraw, y, x2, y2, dw1, dw2, err, lastW;

    timer.tic();
    x = xx.Convolve(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey);
    timer.tac("Convolve: ");

    Mat w(x.getY(), hidden); //init weights
    cout << xCpu->getX() << ", " << xx.getX() << ", " << xCpu->getY() << ", " << hidden << endl;

    if(xCpu->getX() != w.getX() || xCpu->getY() != w.getY())
    {
        w.RandNormal(0.0f, 1.0f/(10*hidden));
        cout << "weight matrix randomized!" << endl;
    }
    else
    {
        w = *xCpu;
        cout << "weight matrix loaded!" << endl;
    }
    //msgG("w", w);
    delete xCpu;


    //w = 0.0f;
    ms("w", w);

    lastW = w;

    cout << endl;
    
    float minErr = FLT_MAX;
    int minIndex = 0;

    bool ONE_ROW = true;


    msgG("xxxxx", x);
    msgG("wwwww", w);

  
    for(int i = 0; i < iterations; ++i)
    {
        //Mat xraw = xx;
        //Mat xraw = xx.Sample(batchSize);

        //x = xraw.Convolve(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey);

        //cout << "x:" << x.getX() << ", y:" << x.getY() << endl;

        //saveGpuMatrix(x, string(argv[1]) + ".convolved");

        //Mat reverse;
        //reverse = x.DeConvolve(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey, normalizer);

        //saveGpuMatrix(reverse, string(argv[1]) + ".reversed");
        //exit(1);

        y = Mult(x, w); // matrixwise -  y.shape = (dataA.x, weights.y) == (dataB.x, dataB.y)
        //msgG("y", y);
        //msgG("y=x*w", y);

        //y = y.Sigmoid();
        //msgG("y", y);

        x2 = Mult(y, w.T());
        //msgG("x2", x2);

        //x2 = x2.Sigmoid();
        //msgG("x2", x2);

        y2 = Mult(x2, w);
        //msgG("y2", y2);

        //y2 = y2.Sigmoid();
        //msgG("y2", y2);

        dw1 = Mult(x.T(), y);
        //msgG("dw1", dw1);
        dw2 = Mult(x2.T(), y2);
        //msgG("dw2", dw2);

        dw1 *= (lSpeed/x.getX());
        dw2 *= (lSpeed/x.getX());

        w = w + dw1;
        w = w - dw2;

        //lastW *= 0.00001;
        //w = w - lastW;

        lastW = w;
        //msgG("w", w);

        ms("w = w + dw", w);

        if(i % 50 == 0 || i+1 == iterations )
        {
            cout << i << ": ";
            float terr = computeError(x, x2);

            cout << "              " << flush;

            if(ONE_ROW)
            {
                cout << "              " << "\r" << flush;
            }
            else
            {
                cout << endl;
            }
            if(terr < minErr)
            {
                minErr = terr;
                minIndex = i;
            }
        }
    }
    cout << endl;

    MatrixCpu res = w;

    msgC("res", res);
    saveMatrix(res, string(argv[1]) + ".weights");

    y = Mult(x, w);
    //y = y.Sigmoid();
    //msgG("y - raw", y);
    //y.Reshape(xx.getX(), transX*transY*hidden);
    msgG("y", y);
    Mat yyy = y.TransformToUpperLayer(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey, xx.getX());
    msgG("trans(y)", yyy);
    MatrixCpu resy = yyy;
    saveMatrix(resy, string(argv[1]) + ".transform");
    Mat zzz = yyy.TransformFromUpperLayer(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey, xx.getX());
    msgG("retranst(trans(y))", zzz);

    //y.Reshape(hidden*xx.getX(), transX*transY);
    //msgG("reshaped(y)", y);
    //MatrixCpu resy = y;
    //saveMatrix(resy, string(argv[1]) + ".transformRaw");

    //exit(1);
    
    //Mat yy = y.T();
    //yy.MakeHardCopy();
    //msgG("transposed(y)", yy);
    //saveMatrix(resy, string(argv[1]) + ".transformRawTransposed");
    
    //saveMatrix(resy, string(argv[1]) + ".transform");
    exit(1);
    x2 = Mult(y, w.T());
    //x2 = x2.Sigmoid();
    Mat reverse, normalizer;
    timer.tic();
    normalizer = x.DeConvolveNormalizer(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey, xx.getX());
    timer.tac("DeConvolveNormalizer: ");
    timer.tic();
    reverse = x2.DeConvolve(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey, normalizer);
    timer.tac("DeConvolve: ");
    MatrixCpu resx = reverse;
    saveMatrix(resx, string(argv[1]) + ".reconstruct");


    cout << "done" << endl << "Min. test error = " << minErr << ", iteration = " << minIndex << endl;

    cout << "done" << endl;*/
    return 0;

}
