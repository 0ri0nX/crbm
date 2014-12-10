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
    cout << "GPU: " << inMsg << ":" << endl;
    MatrixCpu x = inM;
    if(x.getX()*x.getY() > 400)
    {
        cout << x.getX() << " x " << x.getY() << endl;
        cout << "[ " << (x.getData()[0]) << " ... " << (x.getData()[x.getX()*x.getY()-1]) << " ]" << endl;
    }
    else
    {
        x.Save(cout);
    }
    cout << endl;
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

void setIdentity(MatrixCpu &inA)
{
    for(int i = 0; i < inA.getX()*inA.getY(); ++i)
    {
        inA.getData()[i] = 0.0f;
    }

    int m = min(inA.getX(), inA.getY());

    for(int i = 0; i < m; ++i)
    {
        inA.getData()[i*inA.getX() + i] = 1.0f;
    }
}
void setScalar(MatrixCpu &inA, float inValue)
{
    for(int i = 0; i < inA.getX()*inA.getY(); ++i)
    {
        inA.getData()[i] = inValue;
    }
}
void setSequence(MatrixCpu &inA, float start = 0.0f, float increment = 1.0f)
{
    for(int i = 0; i < inA.getX()*inA.getY(); ++i)
    {
        inA.getData()[i] = start + float(i)*increment;
    }
}

#define testParallelAssociativeoperation(matrix, function)\
{\
    Mat mm;\
    mm = matrix.function();\
    ms(#matrix "." #function "()", mm);\
}

int main(int argc, char** argv)
{
    
    /*
    int ix = atoi(argv[1]);
    int iy = atoi(argv[2]);
    int iz = atoi(argv[3]);
    testGpu(ix, iy);
    //testCpu(ix, iy);
    return 0;
    */

   
    if(argc != 4)
    {
        cout << "Too few params!" << endl;
        cout << argv[0] << "x y z" << endl;
        cout << "for (x,y) * (y,z)" << endl;
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

    int ix = atoi(argv[1]);
    int iy = atoi(argv[2]);
    int iz = atoi(argv[3]);

    MatrixCpu *xCpu = new MatrixCpu();
    MatrixCpu *tCpu = new MatrixCpu();
    MatrixCpu *wCpu = new MatrixCpu();

    xCpu->Reset(ix, iy);
    tCpu->Reset(ix, iz);
    wCpu->Reset(iy, iz);

    //setScalar(*xCpu, 3.0);
    setSequence(*xCpu);

    setSequence(*tCpu);

    setSequence(*wCpu, 1.0f, 3.1f);
    //setIdentity(*wCpu);
    //setScalar(*wCpu, -2.0);

    Mat x = *xCpu;
    ms("x", x);
    //Mat xt = x^"T";
    //ms("x^T", xt);
    
    Mat t = *tCpu;
    ms("t", t);

    Mat w = *wCpu;
    ms("w", w);

    delete xCpu;
    delete tCpu;
    delete wCpu;

    if(0)
    {

        //Mat res = x * w;
        //ms("res = x * w", res);
        //Mat res = (x^"T") * w;
        Mat res = x.trans() * w;
        ms("res = (x^\"T\") * w", res);

        //Mat sum;
        //sum = w.Sum();
        //ms("sum = w.Sum()", sum);

        testParallelAssociativeoperation(w, Sum)
        testParallelAssociativeoperation(w, Min)
        testParallelAssociativeoperation(w, Max)
        testParallelAssociativeoperation(w, Multiply)
    }

#define timeIt(x)\
    {\
        //clock_t st = clock();\
        x;\
        //cudaDeviceSynchronize();\
        //clock_t en = clock();\
        //cout << "Duration (" #x "): " << (en - st) / double(CLOCKS_PER_SEC) << " sec" << endl;\
    }


    if(1)
    {

        {
            clock_t st = clock();
            for(int i = 0; i < 1000; ++i)
            {
                t = (x-t);
            }
            cudaDeviceSynchronize();
            clock_t en = clock();
            ms("t", t);
            cout << "Total duration: " << (en - st) / double(CLOCKS_PER_SEC) << " sec" << endl;
        }
        //{
        //    clock_t st = clock();
        //    for(int i = 0; i < 1000; ++i)
        //    {
        //        t = x.TestMinus(t);
        //    }
        //    cudaDeviceSynchronize();
        //    clock_t en = clock();
        //    ms("t", t);
        //    cout << "Total duration: " << (en - st) / double(CLOCKS_PER_SEC) << " sec" << endl;
        //}
    }

    return 0;

}
