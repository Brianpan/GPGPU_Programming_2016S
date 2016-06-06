#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <helper_cuda.h>
#include <cuComplex.h>
#include <cufft.h>

#define SIGSIZE 5
#define SIGDIM 1
#define NBLK 256
#define TIMESLOT 439
//exp i
//https://devtalk.nvidia.com/default/topic/505308/complex-number-exponential-function/
__device__ float angle_trans(const cuComplex& z){
	return atan2(cuCimagf(z), cuCrealf(z));
}

__global__ void fft_polar_angle(cufftComplex *data, float *angle, float *mag, int data_size){
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if(idx >= data_size){
		return;
	}
	//abs of fft
	mag[idx] = cuCabsf(data[idx]);
	//angle of fft
	angle[idx] = angle_trans(data[idx]);
	return;
}

int main(int argc, char **argv)
{	

	// host assign
	cufftComplex *h_signal = (cufftComplex *)malloc(sizeof(cufftComplex)*SIGSIZE*SIGDIM);

	for(int i = 0; i < SIGSIZE*SIGDIM; i ++){
		// h_signal[i].x = rand() / (float) RAND_MAX;
		h_signal[i].x = i;
		h_signal[i].y = 0;
	}
	int data_size = SIGSIZE*SIGDIM;
	int mem_size = sizeof(cufftComplex)*data_size;
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

	//do angle implement in matlab
	float *d_angle, *d_mag;
	checkCudaErrors(cudaMalloc(&d_angle, sizeof(float)*data_size));
	checkCudaErrors(cudaMalloc(&d_mag, sizeof(float)*data_size));

	fft_polar_angle<<<(data_size+NBLK-1)/NBLK, NBLK>>>(d_signal, d_angle, d_mag, data_size);

	float *h_angle = (float *) malloc(sizeof(float)*data_size);
	float *h_mag = (float *) malloc(sizeof(float)*data_size);

	checkCudaErrors(cudaMemcpy(h_angle, d_angle, sizeof(float)*data_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_mag, d_mag, sizeof(float)*data_size, cudaMemcpyDeviceToHost));

	// backward transform
	// printf("---Inverse fft transform --- \n");
	// cufftExecC2C(plan, d_signal, d_signal, 
							   // CUFFT_INVERSE);
	
	cufftComplex *h_inverse_signal = (cufftComplex *) malloc(sizeof(cufftComplex)*SIGDIM*SIGSIZE);
	checkCudaErrors(cudaMemcpy(h_inverse_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

	for(int i = 0; i < SIGSIZE*SIGDIM; i ++){
		if(i%SIGSIZE == 0){
			printf("---- column %d started ---- \n", i/SIGSIZE+1);
		}
		printf("before: %f , %f ; after: %f , %f ; angle: %f ; mag: %f \n",
				h_signal[i].x, h_signal[i].y,
				h_inverse_signal[i].x, h_inverse_signal[i].y, 
				h_angle[i], h_mag[i]);
	}

	free(h_signal);
	// free(h_inverse_signal);
	
	cufftDestroy(plan);
	checkCudaErrors(cudaFree(d_signal));

	cudaDeviceReset();
	
	return 0;
}	