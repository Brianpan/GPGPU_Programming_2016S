NVCC=nvcc
CUDA_INCLUDEPATH=/usr/local/cuda-7.5/include
EXE=a.out

all: lab3.o pgm.o
	$(NVCC) -std=c++11 -arch=sm_30 -O2 main.cu lab3.o pgm.o -o $(EXE)
lab3.o: 
	$(NVCC) -std=c++11 -arch=sm_30 -O2 -c lab3.cu -o lab3.o	
pgm.o:
	gcc -O -c -std=c++11 pgm.cpp -o pgm.o
clean:
	rm a.out lab3.o pgm.o  output.*
exec:
	./a.out img_background.ppm img_target.ppm img_mask.pgm 130 600 ./output.ppm
convert:
	convert output.ppm output.png
