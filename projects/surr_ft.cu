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
#include <curand.h>

#define SIGSIZE 7
#define SIGDIM 2
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

// do p(2:N)=[p1 -flipud(p1)];
__global__ void surr_trans(float *angle, float *ran, int data_size, int sig_size, int half_sig_size){
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if(idx >= data_size){
		return;
	}
	
	int data_col = idx/sig_size;
	int data_idx = idx%sig_size;
	// p(1) is not necessary for changing
	if(data_idx ==0){
		return;
	}
	
	int half_idx;
	//p(2: 2+half-1)
	if(data_idx <= half_sig_size){
		half_idx = (data_idx-1) + data_col*half_sig_size;
		angle[idx] = ran[half_idx];
			
	// -flipup(p1)	
	}else{
		int diff = data_idx - half_sig_size;
		int reverse_data_idx = half_sig_size- diff;
		half_idx = reverse_data_idx + data_col*half_sig_size;
		angle[idx] = -ran[half_idx];
	}

	return;

}

struct pi_mul_trans{
	__host__ __device__ float operator()(const float &ran_val){
		// float pi = 3.14159;
		return 2*3.14159*ran_val;
	}
};

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

	// start parallel surrogate
	int half_col_size = SIGSIZE/2;
	int half_size = half_col_size*SIGDIM;
	float *d_ran_series, *d_temp;
	checkCudaErrors(cudaMalloc(&d_ran_series, sizeof(float)*half_size));
	checkCudaErrors(cudaMalloc(&d_temp, sizeof(float)*half_size));
	if(SIGSIZE%2==0){

	}else{
		//random generator
		curandGenerator_t gen;
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
		curandGenerateUniform(gen, d_ran_series, half_size);

		thrust::device_ptr<float> ran_ptr(d_ran_series);

		//transform rand*2*pi
		thrust::device_ptr<float> temp(d_temp);
		thrust::transform(ran_ptr, ran_ptr+half_size*sizeof(float),
						  temp, pi_mul_trans());
		float *h_ran = (float *) malloc(half_size*sizeof(float));
		float *h_tmp = (float *) malloc(half_size*sizeof(float));
		checkCudaErrors(cudaMemcpy(h_ran, d_ran_series, half_size*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_tmp, d_temp, half_size*sizeof(float), cudaMemcpyDeviceToHost));

		for(int i = 0; i< half_size; i ++){
			printf("ran: %f pi: %f \n", h_ran[i], h_tmp[i]);
		}
		// do column vector trans p(2:N)=[p1 -flipud(p1)];
		surr_trans<<<(data_size+NBLK-1)/NBLK, NBLK>>>(d_angle, d_temp, data_size, SIGSIZE, half_col_size);
		free(h_ran);
		free(h_tmp);
	}

	float *h_angle = (float *) malloc(sizeof(float)*data_size);
	// float *h_mag = (float *) malloc(sizeof(float)*data_size);

	checkCudaErrors(cudaMemcpy(h_angle, d_angle, sizeof(float)*data_size, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(h_mag, d_mag, sizeof(float)*data_size, cudaMemcpyDeviceToHost));

	// backward transform
	// printf("---Inverse fft transform --- \n");
	// cufftExecC2C(plan, d_signal, d_signal, 
							   // CUFFT_INVERSE);
	
	// cufftComplex *h_inverse_signal = (cufftComplex *) malloc(sizeof(cufftComplex)*SIGDIM*SIGSIZE);
	// checkCudaErrors(cudaMemcpy(h_inverse_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

	// for(int i = 0; i < SIGSIZE*SIGDIM; i ++){
	// 	if(i%SIGSIZE == 0){
	// 		printf("---- column %d started ---- \n", i/SIGSIZE+1);
	// 	}
	// 	printf("angle: %f \n", h_angle[i]);
	// 	// printf("before: %f , %f ; after: %f , %f ; angle: %f ; mag: %f \n",
	// 	// 		h_signal[i].x, h_signal[i].y,
	// 	// 		h_inverse_signal[i].x, h_inverse_signal[i].y, 
	// 	// 		h_angle[i], h_mag[i]);
	// }

	free(h_signal);
	// free(h_angle);
	
	// free(h_inverse_signal);
	
	cufftDestroy(plan);
	checkCudaErrors(cudaFree(d_signal));

	cudaDeviceReset();
	
	return 0;
}	