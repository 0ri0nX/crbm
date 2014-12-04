#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>
#include <time.h>

using namespace std;

#include "matrix.h"
#include "utils.h"
#include "crbm.h"

typedef MatrixGpu Mat;

int main(int argc, char** argv)
{
    if(argc != 7 && argc != 8)
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
    Mat xx = *xCpu;

    delete xCpu;
    xCpu = new MatrixCpu();


    CRBM::CRBMLayer *abc = NULL;

    Timer timer;
    if(string(argv[2]) != "-")
    {
        abc = new CRBM::CRBMLayer();
        abc->Load(string(argv[2]));
    }
    else
    {
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

        cout << "Creating RBM-layer ... " << flush;
        abc = new CRBM::CRBMLayer(im_x, im_y, im_z, im_cx, im_cy, im_stridex, im_stridey, hidden);
        abc->setLearningSpeed(lSpeed);
        timer.tac("done ");

    }


    timer.tic();
    abc->LearnBatch(xx, iterations);
    timer.tac("learning duration: ");

    Mat transformed;
    abc->Transform(xx, transformed);

    saveMatrix(transformed, string(argv[1]) + ".transformed");
    abc->Save(string(argv[1]) + ".rbm");

    return 0;
}
