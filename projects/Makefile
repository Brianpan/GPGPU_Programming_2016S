NVCC=nvcc
cufft:
	$(NVCC) -std=c++11 -arch=sm_20 -lcufft -I ./inc/ surr_ft.cu -o surr_ft.o 
clean:
	rm *.o	