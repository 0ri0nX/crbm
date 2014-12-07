#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>

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

void loadMatrix(MatrixCpu &inM, const string& filename, bool inTransposed = false)
{
    cout << "loading [" << filename << "] ... " << endl;
    ifstream f(filename.c_str());
    inM.Load(f, inTransposed);
    f.close();
}

void saveMatrix(MatrixCpu &inM, const string &filename)
{
    cout << "saving [" << filename << "] ... " << endl;
    ofstream f(filename.c_str());
    inM.Save(f);
    f.close();
}


void msgC(char * inMsg, const MatrixCpu &x)
{
    int n = x.getX()*x.getY();
    if(n > 400)
    {
        cout << inMsg << ": " << x.getX() << " x " << x.getY()
             << "[ " << (x.getDataConst()[0]) << ", " << (x.getDataConst()[1]) << " ... " << (x.getDataConst()[n-2]) << ", " << (x.getDataConst()[n-1]) << " ]" << endl;
    }
    else if(n == 1)
    {
        cout  << inMsg << ":[" << x.getDataConst()[0] << "]" << flush;
    }
    else
    {
        cout  << inMsg << ":" << endl;
        x.Save(cout);
        cout << endl;
    }
}

void msgG(char * inMsg, const MatrixGpu &inM)
{
    MatrixCpu x = inM;
    msgC(inMsg, x);
}


void ms(char * inMsg, const MatrixGpu &inM)
{
    //msgG(inMsg, inM);
}


void testGpu(int x, int y)
{
    typedef MatrixGpu M;
    typedef MatrixCpu MC;
    
    cout << "GPU -----------" << endl;

    MC ac(x, y);
    for(int i = 0; i < ac.getX()*ac.getY(); ++i)
    {
        ac.getData()[i] = float(i);
    }
    M a = ac;
    msgG("a - init", a);

    //a = 11.0f;
    //msgG("a=11.0f", a);

    M b = a.AbsSum();
    msgG("b=a.AbsSum()", b);

    MC cc(y, 3);
    for(int i = 0; i < cc.getX()*cc.getY(); ++i)
    {
        cc.getData()[i] = 0.0f;
    }
    cc.getData()[0] = 1.0f;
    cc.getData()[y+1] = 1.0f;
    cc.getData()[2*y+2] = 1.0f;
    M c = cc;
    msgG("c", c);

    M d = Mult(a, c);
    msgG("d=a*c", d);

    
}

void testCpu(int x, int y)
{
    typedef MatrixCpu M;

    cout << "CPU -----------" << endl;
    
    M a(x, y);
    msgC("a - init", a);

    //a = 11.0f;
    for(int i = 0; i < a.getX()*a.getY(); ++i)
    {
        a.getData()[i] = 11;
    }
    msgC("a=11.0f", a);

    M b(1,1);
    float sum = 0.0f;
    for(int i = 0; i < a.getX()*a.getY(); ++i)
    {
        sum += a.getData()[i];
    }
    b.getData()[0] = sum;
    msgC("sum=a.AbsSum()", b);
}

//const float x1[] = {1.0f, 0.0f, 0.0f};
//const float t1[] = {1.0f, 0.0f};
//
//const float x2[] = {1.0f, 0.0f, 1.0f};
//const float t2[] = {1.0f, 0.0f};
//
//const float x3[] = {1.0f, 1.0f, 0.0f};
//const float t3[] = {1.0f, 0.0f};
//
//const float x4[] = {1.0f, 1.0f, 1.0f};
//const float t4[] = {0.0f, 1.0f};

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
    if(argc < 5)
    {
        cout << "Too few params!" << endl;
        cout << argv[0] << " <gpu-id> <reconstruct|transform> input-vector-file weights-file1 [weights-file2] ..." << endl;
        exit(1);
    }

    cublasStatus_t stat;
    cublasHandle_t handle;

    cudaSetDevice(atoi(argv[1]));
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    string computationType = argv[2];
    if(computationType != "reconstruct" && computationType != "transform")
    {
        cout << "Unsupported computation type: [" << computationType << "]" << endl;
        exit(1);
    }

    MatrixCpu *xCpu = new MatrixCpu();

    loadMatrix(*xCpu, argv[3]);

    int rows = xCpu->getX();
    int cols = xCpu->getY();

    Mat xx = *xCpu;
    msgG("loaded", xx);

    std::vector<Mat*> weights;

    for(int i = 4; i < argc; ++i)
    {
        loadMatrix(*xCpu, string(argv[i]));
        cout << i-3 << ". ";
        msgC("weights", *xCpu);
        Mat *w = new Mat(*xCpu);
        weights.push_back(w);
    }

    Mat y;

    for(int i = 0; i < weights.size(); ++i)
    {
        cout << "Transforming with weights " << i+1 << endl;
        y = Mult(xx, *(weights[i]));
        xx = y;
    }

    if(computationType == "transform")
    {
        MatrixCpu resx = xx;
        saveMatrix(resx, string(argv[3]) + ".transform");
        exit(1);
    }
    
    for(int i = weights.size() - 1; i >= 0; --i)
    {
        cout << "Reconstructing with weights " << i+1 << endl;
        y = Mult(xx, (weights[i])->T());
        xx = y;
    }

    if(computationType == "reconstruct")
    {
        MatrixCpu resx = xx;
        saveMatrix(resx, string(argv[3]) + ".reconstruct");
        exit(1);
    }

    cout << "done" << endl;
    return 0;

}
