#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include <cufft.h>

#define SIGSIZE 5
#define SIGDIM 3
int main(int argc, char **argv)
{	

	// host assign
	cufftComplex *h_signal = (cufftComplex *)malloc(sizeof(cufftComplex)*SIGSIZE*SIGDIM);

	for(int i = 0; i < SIGSIZE*SIGDIM; i ++){
		h_signal[i].x = rand() / (float) RAND_MAX;
		h_signal[i].y = 0;
	}

	int mem_size = sizeof(cufftComplex)*SIGSIZE*SIGDIM;
	cufftComplex *d_signal;
	checkCudaErrors(cudaMalloc((void **) &d_signal, mem_size));

	checkCudaErrors(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

	//cufft
	cufftHandle plan;
	if (cufftPlan1d(&plan, SIGSIZE, CUFFT_C2C, SIGDIM) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
	}

	//forward transform
	printf("---Transform fft--- \n");
	cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);

	//backward transform
	printf("---Inverse fft transform --- \n");
	cufftExecC2C(plan, d_signal, d_signal, 
							   CUFFT_INVERSE);
	
	cufftComplex *h_inverse_signal = (cufftComplex *) malloc(sizeof(cufftComplex)*SIGDIM*SIGSIZE);
	checkCudaErrors(cudaMemcpy(h_inverse_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

	for(int i = 0; i < SIGSIZE*SIGDIM; i ++){
		if(i%SIGSIZE == 0){
			printf("---- column %d started ---- \n", i/SIGSIZE+1);
		}
		printf("before: %f , %f ; after: %f , %f \n",
				h_signal[i].x, h_signal[i].y,
				h_inverse_signal[i].x/(float)SIGSIZE, h_inverse_signal[i].y/(float)SIGSIZE);
	}

	free(h_signal);
	free(h_inverse_signal);
	
	cufftDestroy(plan);
	checkCudaErrors(cudaFree(d_signal));

	cudaDeviceReset();
	
	return 0;
}	