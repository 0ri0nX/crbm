all: crbm rbm rbmCompute
#all: rbm linearCombiner linearCombinerDynamic compute

%.o: %.cu matrix.h Makefile
	nvcc -c -o $@ -O3 -D CUDA -g $<

test: test.cu matrix.h
	nvcc -o test -D CUDA test.cu -lcublas -lcurand -D DEBUG

crbm: matrix.h crbm.o
	nvcc -o crbm -O3 -D CUDA -lcuda -lcublas -lcurand -lcudart crbm.o -g

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
