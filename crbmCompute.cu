#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <string>
#include <time.h>
#include <csignal>

using namespace std;

#include "matrix.h"
#include "utils.h"
#include "crbm.h"

typedef MatrixGpu Mat;

using namespace YAMATH;

int main(int argc, char** argv)
{
    if(argc < 5)
    {
        cout << "Too few params!" << endl;
        cout << argv[0] << " <gpu-id> <reconstruct|transform> input-vector-file crbm-file1 [crbm-file2] ..." << endl;
        exit(1);
    }

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    cudaSetDevice(atoi(argv[1]));
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

    std::vector<CRBM::CRBMLayer*> layers;

    for(int i = 4; i < argc; ++i)
    {
        cout << i-3 << ". ";
        CRBM::CRBMLayer *l = new CRBM::CRBMLayer();
        l->Load(string(argv[i]));
        layers.push_back(l);
    }

    Mat y;

    for(int i = 0; i < layers.size(); ++i)
    {
        cout << "Transforming with layer " << i+1 << endl;
        layers[i]->Transform(xx, y);
        xx = y;
    }

    if(computationType == "transform")
    {
        saveMatrix(xx, string(argv[3]) + ".transformed");
        return 0;
    }
    
    for(int i = layers.size() - 1; i >= 0; --i)
    {
        cout << "Reconstructing with layer " << i+1 << endl;
        layers[i]->Reconstruct(xx, y);
        xx = y;
    }

    if(computationType == "reconstruct")
    {
        saveMatrix(xx, string(argv[3]) + ".reconstruct");
        return 0;
    }

    return 0;

}
