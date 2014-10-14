#all: test linearCombiner compute
all: linearCombiner linearCombinerDynamic nonLinCombinerDynamic


test: test.cu matrix.h
	nvcc -o test -D CUDA test.cu -lcublas -lcurand -D DEBUG

linearCombiner: linearCombiner.o matrix.h linearCombiner.cu
	nvcc -o linearCombiner -D DEBUG -D CUDA -lcuda -lcublas -lcurand -lcudart linearCombiner.o

linearCombiner.o: linearCombiner.cu matrix.h
	nvcc -c linearCombiner.cu -D DEBUG -D CUDA

linearCombinerDynamic: linearCombinerDynamic.o matrix.h linearCombinerDynamic.cu
	nvcc -o linearCombinerDynamic -D DEBUG -D CUDA -lcuda -lcublas -lcurand -lcudart linearCombinerDynamic.o

linearCombinerDynamic.o: linearCombinerDynamic.cu matrix.h
	nvcc -c linearCombinerDynamic.cu -D DEBUG -D CUDA

nonLinCombinerDynamic: nonLinCombinerDynamic.o matrix.h nonLinCombinerDynamic.cu
	nvcc -o nonLinCombinerDynamic -D DEBUG -D CUDA -lcuda -lcublas -lcurand -lcudart nonLinCombinerDynamic.o

nonLinCombinerDynamic.o: nonLinCombinerDynamic.cu matrix.h
	nvcc -c nonLinCombinerDynamic.cu -D DEBUG -D CUDA

compute: compute.cu matrix.h
	nvcc -o compute -D CUDA compute.cu -lcublas -lcurand

clean:
	rm -f linearCombiner test compute
