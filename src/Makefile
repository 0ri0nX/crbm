all: crbm crbmOld
#all: rbm linearCombiner linearCombinerDynamic compute

%.o: %.cu matrix.h Makefile utils.h crbm.h
	nvcc --gpu-architecture=sm_35  -c -o $@ -O3 -D CUDA -g $<

%.o: %.cpp
	g++ -std=c++0x -c -o $@ -O3 -g $<

test: test.cu matrix.h
	nvcc -o test -D CUDA test.cu -lcublas -lcurand -D DEBUG

crbm: matrix.h utils.h crbm.h crbm.o Makefile utils.h matrixCpu.o
	nvcc --gpu-architecture=sm_35 -o crbm -O3 -D CUDA -lcuda -lcublas -lcurand -lcudart crbm.o matrixCpu.o -g

crbmCompute: matrix.h utils.h crbm.h crbmCompute.o Makefile utils.h matrixCpu.o
	nvcc -Xcompiler "-std=c++0x" --gpu-architecture=sm_35 -o crbmCompute -O3 -D CUDA -lcuda -lcublas -lcurand -lcudart crbmCompute.o matrixCpu.o -g

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
	rm -f linearCombiner test compute
