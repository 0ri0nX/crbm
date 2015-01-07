#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>
#include <time.h>
#include <csignal>
#include <iomanip>
#include <cmath>

using namespace std;

#include "matrix.h"
#include "crbmGpu.h"

using namespace YAMATH;

typedef MatrixGpu Mat;

string getName(const string &inPrefix, int inIdx, int inTotal)
{
    stringstream s;
    int w = ceil(log10(inTotal));
    s << inPrefix.c_str() << setfill('0') << setw(w) << inIdx;

    return s.str();
}

int main(int argc, char** argv)
{
    if(argc < 5)
    {
        cout << "Too few params!" << endl;
        cout << argv[0] << " <gpu-id> <reconstruct|transform> input-vector-file crbm-file1 [crbm-file2] ..." << endl;
        exit(1);
    }


    cudaSetDevice(atoi(argv[1]));
    cublasStatus_t stat;
    cublasHandle_t handle;

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

    int batchSize = 500;

    cout << "Maximal batch size: " << batchSize << endl;

    MatrixCpu xCpu;

    Timer timer;
    t_index cols = 0;
    t_index rows = 0;
    int loadVersion = -1;

    //data-file
    std::ifstream fdata(argv[3]);

    xCpu.LoadHeader(fdata, loadVersion, rows, cols);

    int batchNum = (rows - 1) / batchSize + 1;

    std::vector<CRBM::CRBMLayerGpu*> layers;

    for(int i = 4; i < argc; ++i)
    {
        cout << i-3 << ". ";
        CRBM::CRBMLayerGpu *l = new CRBM::CRBMLayerGpu();
        l->Load(string(argv[i]));
        layers.push_back(l);
    }

    int resSize = -1;
    string outFilename = string(argv[3]);

    if(computationType == "transform")
    {
        outFilename += ".transformed";

        int outX, outY;
        layers.back()->getConvolutionPatchesNumber(outX, outY);
        resSize = outX*outY*layers.back()->s().hidden;
    }
    else
    {
        outFilename += ".reconstruct";
        resSize = cols;
    }

    cout << "Saving into: [" << outFilename << "]" << endl;

    ofstream f(outFilename.c_str());

    int saveVersion = -1;
    if(computationType == "reconstruct")
    {
        saveVersion = 0;
    }
    if(computationType == "transform")
    {
        saveVersion = 2;
    }

    MatrixCpu::SaveHeader(f, rows, resSize, saveVersion);
    //f << rows << " " << resSize << endl;

    MatrixCpu xxCpu;
    Mat xx, xxTrans;
    MatrixCpu tmpxx;
    timer.tic();

    for(int batch = 1; batch <= batchNum; ++batch)
    {
        //int a = batch*batchSize;
        //int b = min((batch+1)*batchSize, rows);

        cout << batch << " / " << batchNum << endl;

        timer.tic();

        t_index actBatchSize = (batch != batchNum) ? batchSize : (rows - (batchNum-1)*batchSize);

        xxCpu.LoadBatch(fdata, false, loadVersion, actBatchSize, cols, "");
        xx = xxCpu;

        //xx = xCpu->SubMatrix(a, 0, b, cols);

        //xx = xCpu->SubMatrix(0, a, cols, b);

        //transposition needed
        //xx.Transpose();
        //xx.MakeHardCopy();

        timer.tac("   selected: ");

        Mat y;

        for(int i = 0; i < layers.size(); ++i)
        {
            timer.tic();
            cout << "   Transforming with layer " << i+1 << flush;
            layers[i]->Transform(xx, y);
            timer.tac(" ");
            xx = y;
        }

        if(computationType == "transform")
        {
            timer.tic();
            tmpxx = xx;
            tmpxx.Save(f, false, saveVersion);
            timer.tac("   saved: ");
            continue;
        }
        
        for(int i = layers.size() - 1; i >= 0; --i)
        {
            timer.tic();
            cout << "   Reconstructing with layer " << i+1 << flush;
            layers[i]->Reconstruct(xx, y);
            timer.tac(" ");
            xx = y;
        }

        if(computationType == "reconstruct")
        {
            timer.tic();
            tmpxx = xx;
            tmpxx.Save(f, false, saveVersion);
            timer.tac("   saved: ");
            continue;
        }
    }

    f.close();
    cout << "Saved into: [" << outFilename << "]" << endl;

    return 0;
}
