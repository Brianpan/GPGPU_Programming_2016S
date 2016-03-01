NVCC=nvcc
CUDA_INCLUDEPATH=/usr/local/cuda-7.5/include
all: Makefile
	$(NVCC) main.cu -std=c++11 -o main.o
exec: test.txt main.o 
	./main.o test.txt
clean:
	rm -f *.o

