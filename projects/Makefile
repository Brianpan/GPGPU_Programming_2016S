NVCC=nvcc
cufft:
	$(NVCC) -std=c++11 -arch=sm_30 -lcufft -lcurand -I ./inc/ surr_ft.cu -o surr_ft.o 
clean:
	rm *.o
curand:
	$(NVCC) -std=c++11 -arch=sm_30 sort2d.cu -o sort2d.o	