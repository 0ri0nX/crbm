all: libcrbmcomputercpu.so
#all: libcrbmcomputergpu.so libcrbmcomputercpu.so

libcrbmcomputergpu.so: crbmComputerBinding.o crbmComputerGpu.o matrixCpu.o crbmComputer.h crbmComputer.cu crbmComputerBinding.h
	g++ -L/usr/local/cuda-5.5/lib64 -shared -fPIC -o $@ crbmComputerBinding.o crbmComputerGpu.o matrixCpu.o -lcuda -lcublas -lcurand -lcudart

crbmComputerGpu.o: crbmComputer.cu crbmComputer.h
	nvcc --gpu-architecture=sm_35 -Xcompiler '-fPIC' -D CUDA -c -o $@ crbmComputer.cu -g

libcrbmcomputercpu.so: crbmComputerBinding.o crbmComputerCpu.o matrixCpu.o crbmComputer.h crbmComputer.cu crbmComputerBinding.h
	g++ -L/usr/local/cuda-5.5/lib64 -shared -fPIC -o $@ crbmComputerBinding.o crbmComputerCpu.o matrixCpu.o -lblas -lrt

crbmComputerCpu.o: crbmComputer.cu crbmComputer.h
	g++ -std=c++0x -fPIC -c -o $@ crbmComputer.cpp -g

matrixCpu.o: ../matrixCpu.cpp ../matrixCpu.h
	g++ -std=c++0x -fPIC -c -o $@ $< -g -D SQUEEZE

%.o: %.cpp
	g++ -std=c++0x -fPIC -c -o $@ $^ -g -Wall

%.o: %.cu
	nvcc --gpu-architecture=sm_35 -Xcompiler '-fPIC' -D CUDA -c -o $@ $^ -g


clean:
	rm -f *.o libcrbmcomputergpu.so libcrbmcomputercpu.so
