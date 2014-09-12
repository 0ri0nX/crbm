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

void msgG(char * inMsg, const MatrixGpu &inM)
{
    MatrixCpu x = inM;
    if(x.getX()*x.getY() > 400)
    {
        cout << "GPU: " << inMsg << ":" << endl;
        cout << x.getX() << " x " << x.getY() << endl;
        cout << "[ " << (x.getData()[0]) << " ... " << (x.getData()[x.getX()*x.getY()-1]) << " ]" << endl;
        cout << endl;
    }
    else if(x.getX()*x.getY() == 1)
    {
        cout << "GPU: " << inMsg << ":" << x.getData()[0] << flush;
    }
    else
    {
        cout << "GPU: " << inMsg << ":" << endl;
        x.Save(cout);
        cout << endl;
    }
}

void msgC(char * inMsg, const MatrixCpu &inM)
{
    cout << "CPU: " << inMsg << ":" << endl;
    const MatrixCpu &x = inM;
    if(x.getX()*x.getY() > 100)
    {
        cout << x.getX() << " x " << x.getY() << endl;
        cout << "[ " << (x.getDataConst()[0]) << " ... " << (x.getDataConst()[x.getX()*x.getY()-1]) << " ]" << endl;
    }
    else
    {
        x.Save(cout);
    }
    cout << endl;
}

void ms(char * inMsg, const MatrixGpu &inM)
{
    msgG(inMsg, inM);
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

    M d = a*c;
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


int main(int argc, char** argv)
{
    if(argc != 4)
    {
        cout << "Too few params!" << endl;
        cout << argv[0] << " input-vector-file weights-file output-file" << endl;
        exit(1);
    }

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    typedef MatrixGpu Mat;

    MatrixCpu *xCpu = new MatrixCpu();
    MatrixCpu *wCpu = new MatrixCpu();

    loadMatrix(*xCpu, argv[1]);
    Mat x = *xCpu;
    ms("x", x);

    loadMatrix(*wCpu, argv[2]);
    Mat w = *wCpu;
    ms("w", w);

    delete xCpu;
    delete wCpu;

    Mat y;
    y = x * w; // matrixwise -  y.shape = (dataA.x, weights.y) == (dataB.x, dataB.y)
    ms("y=x*w", y);

    MatrixCpu res = y;

    msgC("res=", res);
    saveMatrix(res, argv[3]);
    //res.Save(cout);

    cout << "done" << endl;
    return 0;

}
