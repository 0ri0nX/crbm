#all: test linearCombiner compute
all: linearCombiner

test: test.cu matrix.h
	nvcc -o test -D CUDA test.cu -lcublas -lcurand -D DEBUG

linearCombiner: linearCombiner.o matrix.h linearCombiner.cu
	nvcc -o linearCombiner -D DEBUG -D CUDA -lcuda -lcublas -lcurand -lcudart linearCombiner.o
#program -L/usr/local/cuda/lib64 -lcuda

linearCombiner.o: linearCombiner.cu matrix.h
	nvcc -c linearCombiner.cu -D DEBUG -D CUDA

compute: compute.cu matrix.h
	nvcc -o compute -D CUDA compute.cu -lcublas -lcurand

clean:
	rm -f linearCombiner test compute
