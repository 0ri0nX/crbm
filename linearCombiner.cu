#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>

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



using namespace YAMATH;

void loadMatrix(MatrixCpu &inM, char* filename, bool inTransposed = false)
{
    cout << "loading [" << filename << "] ... " << endl;
    ifstream f(filename);
    inM.Load(f, inTransposed);
    f.close();
}

void saveMatrix(MatrixCpu &inM, char* filename)
{
    cout << "saving [" << filename << "] ... " << endl;
    ofstream f(filename);
    inM.Save(f);
    f.close();
}

void ms(const MatrixCpu &inM)
{
    cout << "x:" << inM.getX() << ", y:" << inM.getY() << endl;
    //inM.Save(cout);
}


int main(int argc, char** argv)
{
    if(argc != 4)
    {
        cout << "Too few params!" << endl;
        cout << argv[0] << "image-feature-matrix w2vec-matrix output-weights" << endl;
        exit(1);
    }

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    MatrixCpu *xCpu = new MatrixCpu();
    MatrixCpu *tCpu = new MatrixCpu();
    loadMatrix(*xCpu, argv[1], true);
    ms(*xCpu);
    loadMatrix(*tCpu, argv[2]);
    ms(*tCpu);

    cout << "Copy to GPU ..." << endl;
    MatrixGpu x = *xCpu;
    MatrixGpu t = *tCpu;

    delete xCpu;
    delete tCpu;

    MatrixGpu w(x.getX(), t.getY()); //init weights
    w.Rand();

    //w = x * t;

    //float alpha = 1.0f;
    //float beta = 0.0f;

    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //        x.getX(), t.getY(), x.getY(),
    //        &alpha, x.getData(), x.getX(), t.getData(), t.getX(), &beta, w.getData(), w.getX());


    /*
    for(int i = 0; i < 1000; ++i)
    {
        dev::Matrix y = x * w; // matrixwise -  y.shape = (dataA.x, weights.y) == (dataB.x, dataB.y)
        dev::Matrix e = 0.5f*(t - y)^2; //yDiff.shape = dataB.shape

        if(i % 10 == 0)
        {
            float ee = dev::sumSquared(e); // ee = sum(e^2) elementwise squared sum
            cout << "error:" << ee << endl;
            if(ee < 0.001f)
            {
                break;
            }
        }

        //(t - y)dFi*xi => (t - y)*xi
        dev::Matrix dW = x.trans * e; // == (y - dataB)*dataA ; // elementwise
        
        w += alpha*dW;
    }
    */

    MatrixCpu res = w;

    ms(res);
    saveMatrix(res, argv[3]);
    //res.Save(cout);

    cout << "done" << endl;

    return 0;
}
