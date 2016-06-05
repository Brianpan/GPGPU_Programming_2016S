NVCC=nvcc
f1: 
	$(NVCC) -arch=sm_20 f1.cu -o f1.out
f2:
	$(NVCC) -arch=sm_20 f2.cu -o f2.out
m:
	$(NVCC) -arch=sm_50 -ptx matrix.cu  -o m.ptx
rm:
	rm *.out *.ptx	