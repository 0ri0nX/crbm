all: libcrbmcomputer.so
#all: rbm linearCombiner linearCombinerDynamic compute

libcrbmcomputer.so: crbmComputerBinding.o crbmComputer.o
	g++ -L/usr/local/cuda-5.5/lib64 -shared -fPIC -o $@ $^ -lcuda -lcublas -lcurand -lcudart
%.o: %.cpp
	g++ -fPIC -c -o $@ $^ -g
%.o: %.cu
	nvcc --gpu-architecture=sm_35 -Xcompiler '-fPIC' -D CUDA -c -o $@ $^ -g

clean:
	rm -f libcrbmcomputer.so *.o
