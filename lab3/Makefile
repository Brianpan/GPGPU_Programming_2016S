NVCC=nvcc
CUDA_INCLUDEPATH=/usr/local/cuda-7.5/include
EXE=ans.exe

all: lab3.o pgm.o
	$(NVCC) -std=c++11 -arch=sm_30 -O2 main.cu lab3.o pgm.o -o $(EXE)
lab3.o: 
	$(NVCC) -std=c++11 -arch=sm_30 -O2 -c lab3.cu -o lab3.o	
pgm.o:
	gcc -O -c -std=c++11 pgm.cpp -o pgm.o
clean:
	rm ans.exe lab3.o pgm.o  result.*
exec:
	./ans.exe img_background.ppm img_target.ppm img_mask.pgm 100 100 ./result.ppm
convert:
	convert result.ppm result.png
