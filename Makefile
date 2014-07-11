all: linearCombiner

linearCombiner: linearCombiner.cu matrix.h
	nvcc -o linearCombiner -D CUDA linearCombiner.cu -lcublas -lcurand

clean:
	rm -f linearCombiner
