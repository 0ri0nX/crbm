all: test linearCombiner

test: test.cu matrix.h
	nvcc -o test -D CUDA test.cu -lcublas -lcurand -D DEBUG

linearCombiner: linearCombiner.cu matrix.h
	nvcc -o linearCombiner -D CUDA linearCombiner.cu -lcublas -lcurand -D DEBUG

clean:
	rm -f linearCombiner test
