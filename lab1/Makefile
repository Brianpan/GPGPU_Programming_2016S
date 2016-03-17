NVCC=nvcc
CUDA_INCLUDEPATH=/usr/local/cuda-7.5/include
EXE=ans.exe
main: counting.o
	$(NVCC) -std=c++11 -arch=sm_30 -O2 main.cu counting.o -o $(EXE)
counting.o:
	$(NVCC) -std=c++11 -arch=sm_30 -O2 -c counting.cu -o counting.o

clean:
	rm -f counting.o ans.exe

