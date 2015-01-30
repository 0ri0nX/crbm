#all: crbmComputeCpu
all: crbmComputeGpu crbmGpu crbmComputeCpu
#all: crbm crbmComputeGpu
#all: rbm linearCombiner linearCombinerDynamic compute

%.o: %.cu matrix.h matrixCpu.h Makefile utils.h crbm.h crbmGpu.h
	#nvcc --gpu-architecture=sm_35 -c -o $@ -D CUDA -g $<
	nvcc --gpu-architecture=sm_35 -c -o $@ -O3 -D CUDA -g $<

%.o: %.cpp crbmCpu.h crbm.h Makefile matrix.h matrixCpu.h utils.h
	#g++ -std=c++0x -c -o $@ -g $<
	g++ -std=c++0x -c -o $@ -O3 -g $<

test: test.cu matrix.h
	nvcc -o test -D CUDA test.cu -lcublas -lcurand -D DEBUG

crbmComputeCpu: matrixCpu.h utils.h crbm.h crbmCpu.h crbmComputeCpu.o Makefile matrixCpu.o utils.o
	g++ -std=c++0x -o crbmComputeCpu -O3 crbmComputeCpu.o matrixCpu.o utils.o -lrt -lblas -g

crbmGpu: matrix.h utils.h utilsGpu.h crbm.h crbmGpu.h crbmGpu.o Makefile matrixCpu.o utils.o
	#nvcc --gpu-architecture=sm_35 -o crbmGpu -D CUDA -lcuda -lcublas -lcurand -lcudart crbmGpu.o matrixCpu.o utils.o -lblas -g
	nvcc --gpu-architecture=sm_35 -o crbmGpu -O3 -D CUDA -lcuda -lcublas -lcurand -lcudart crbmGpu.o matrixCpu.o utils.o -lblas -g

crbmComputeGpu: matrix.h utils.h utilsGpu.h crbm.h crbmGpu.h crbmComputeGpu.o Makefile matrixCpu.o utils.o
	nvcc -Xcompiler "-std=c++0x" --gpu-architecture=sm_35 -o crbmComputeGpu -O3 -D CUDA -lcuda -lcublas -lcurand -lcudart -lblas crbmComputeGpu.o matrixCpu.o utils.o -g

crbmOld: matrix.h utils.h crbm.h crbmOld.o Makefile utils.h
	nvcc --gpu-architecture=sm_35 -o crbmOld -O3 -D CUDA -lcuda -lcublas -lcurand -lcudart crbmOld.o -g

rbm: rbm.o matrix.h rbm.cu
	nvcc -o rbm -O3 -D CUDA -lcuda -lcublas -lcurand -lcudart rbm.o
#	nvcc -o rbm -D DEBUG -D CUDA -lcuda -lcublas -lcurand -lcudart rbm.o

rbm.o: rbm.cu matrix.h
	nvcc -c rbm.cu -O3 -D CUDA
#	nvcc -c rbm.cu -D DEBUG -D CUDA

rbmCompute: rbmCompute.o matrix.h rbmCompute.cu
	nvcc -o rbmCompute -O3 -D CUDA -lcuda -lcublas -lcurand -lcudart rbmCompute.o

rbmCompute.o: rbmCompute.cu matrix.h
	nvcc -c rbmCompute.cu -O3 -D CUDA

linearCombiner: linearCombiner.o matrix.h linearCombiner.cu
	nvcc -o linearCombiner -D DEBUG -D CUDA -lcuda -lcublas -lcurand -lcudart linearCombiner.o

linearCombiner.o: linearCombiner.cu matrix.h
	nvcc -c linearCombiner.cu -D DEBUG -D CUDA

linearCombinerDynamic: linearCombinerDynamic.o matrix.h linearCombinerDynamic.cu
	nvcc -o linearCombinerDynamic -D DEBUG -D CUDA -lcuda -lcublas -lcurand -lcudart linearCombinerDynamic.o

linearCombinerDynamic.o: linearCombinerDynamic.cu matrix.h
	nvcc -c linearCombinerDynamic.cu -D DEBUG -D CUDA

compute: compute.cu matrix.h
	nvcc -o compute -D CUDA compute.cu -lcublas -lcurand

clean:
	rm -f crbmComputeGpu crbmGpu crbmComputeCpu *.o
