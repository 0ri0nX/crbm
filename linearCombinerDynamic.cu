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

void computeError(Mat &inW, Mat &inInp, Mat &inOut)
{
    //cout << "inW:" << inW.getX() << " x " << inW.getY() << endl;
    //cout << "inInp:" << inInp.getX() << " x " << inInp.getY() << endl;
    Mat r, r2, r3;
    //msgG("www=", inW);
    //msgG("iiinp=", inInp);
    //msgG("ooout=", inOut);
    r = Mult(inInp, inW);
    //msgG("r=", r);
    r2 = r - inOut;
    //msgG("r2=", r2);
    r2 ^= 2.0f;
    //msgG("r2=", r2);
    r3 = r2.AbsSum();
    //msgG("r3=", r3);
    r3 *= 1.0f / inInp.getX();
    //msgG("r3=", r3);

    msgG("abssum2", r3);
}

int main(int argc, char** argv)
{
    if(argc != 5)
    {
        cout << "Too few params!" << endl;
        cout << argv[0] << " input-vector-file target-vector-file output-weights-file learning-speed" << endl;
        exit(1);
    }

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    float lSpeed = atof(argv[4]);

    MatrixCpu *xxCpu = new MatrixCpu();
    MatrixCpu *ttCpu = new MatrixCpu();

    loadMatrix(*xxCpu, argv[1]);
    msgC("matrix: ", *xxCpu);
    loadMatrix(*ttCpu, argv[2]);
    msgC("matrix: ", *ttCpu);

    int rows = xxCpu->getX();
    int cols = xxCpu->getY();
    int colsT = ttCpu->getY();

    int fract = rows - rows/5;

    MatrixCpu *xCpu = new MatrixCpu(xxCpu->SubMatrix(0, 0, fract, cols));
    MatrixCpu *xCpuTe = new MatrixCpu(xxCpu->SubMatrix(fract, 0, rows, cols));

    MatrixCpu *tCpu = new MatrixCpu(ttCpu->SubMatrix(0, 0, fract, colsT));
    MatrixCpu *tCpuTe = new MatrixCpu(ttCpu->SubMatrix(fract, 0, rows, colsT));

    delete xxCpu;
    delete ttCpu;

    Mat x = *xCpu;
    Mat t = *tCpu;

    Mat xTe = *xCpuTe;
    Mat tTe = *tCpuTe;

    delete xCpu;
    delete tCpu;

    delete xCpuTe;
    delete tCpuTe;


    Mat w(x.getY(), t.getY()); //init weights
    //w.Rand();
    w = 0.0f;

    //learning speed matrix
    Mat ls(x.getY(), t.getY());
    ls = lSpeed;

    Mat lastDir(x.getY(), t.getY());
    lastDir = 0.0f;

    Mat y, e, suma, dw, dty, lsModUp, lsModDown, actDir, lsModMin;

    cout << endl;

    
    for(int i = 0; i < 100000; ++i)
    {
        y = Mult(x, w); // matrixwise -  y.shape = (dataA.x, weights.y) == (dataB.x, dataB.y)
        //msgG("y=x*w", y);

        dty = t - y;
        //msgG("dty=t-y", dty);

        if(i % 1 == 0)
        {
            cout /*<< "\r"*/ << i << ": ";
            computeError(w, x, t);
            computeError(w, xTe, tTe);
            cout << endl;
        }

        dw = Mult(x.T(), dty);
        ms("dw=x^t * dty", dw);

        actDir = dw;
        lsModUp = Mat(lastDir*actDir) >= 0.0f;
        lsModDown = lsModUp <= 0.0f;
        lsModMin = ls < (lSpeed*0.0001f);
        lsModUp *= 1.1f;
        lsModDown *= 0.5f;
        lsModMin *= lSpeed*0.0001f;

        //msgG("dir up      ", lsModUp);
        //msgG("dir down    ", lsModDown);

        ls = ls * Mat(lsModUp + lsModDown);
        //ls = ls * lsMod;//Mat(Mat(Mat(lsMod>=0)*1.1f) + Mat(lsMod<0)*0.5f);
        ls = ls + lsModMin;
        //msgG("speed matrix", ls);

        lastDir = actDir;
        
        dw = dw * ls;
        ms("dw*= lSpeed", dw);

        w = w + dw;
        ms("w = w + dw", w);
    }    

    MatrixCpu res = w;

    msgC("res=", res);
    saveMatrix(res, argv[3]);

    cout << "done" << endl;
    return 0;

}
