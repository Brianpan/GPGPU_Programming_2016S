NVCC=nvcc
CUDA_INCLUDEPATH=/usr/local/cuda-7.5/include
EXE=ans.exe

all: lab2.o
	$(NVCC) -std=c++11 -arch=sm_30 -O2 main.cu lab2.o -o $(EXE)
lab2.o: 
	$(NVCC) -std=c++11 -arch=sm_30 -O2 -c lab2.cu -o lab2.o	
transfer:
	avconv -i result.y4m result.mkv
clean:
	rm result.* ans.exe lab2.o		
