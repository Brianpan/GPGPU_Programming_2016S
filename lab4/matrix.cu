#include <cstdio>
#include <cstdlib>

static const int DIM = 128;
__global__ void Normalize128(float *data, const int N) {
  float tmp[DIM];
  float norm1 = 0;
  float *start = data + threadIdx.x + blockIdx.x*blockDim.x;
  #pragma unroll
  for (int i = 0; i < DIM; ++i) {
    tmp[i] = *start;
    norm1 += abs(tmp[i]);
    start += N;
  }
  float norm1_inv = 1.0f / norm1;
  start = data + threadIdx.x + blockIdx.x*blockDim.x;
  #pragma unroll
  for (int i = 0; i < DIM; ++i) {
    // const int idx = i*N;
    *start = (tmp[i]) * norm1_inv;
    start += N;
  }
}

int main(){
  float *h_ran, *d_ran;
  h_ran = (float *)malloc(sizeof(float)*128);
  for(int i = 0 ; i < 128 ; i ++){
    h_ran[i] = 2;
  }
  cudaMalloc(&d_ran, sizeof(float)*128);
  cudaMemcpy(d_ran, h_ran, sizeof(float)*128, cudaMemcpyHostToDevice);
  
  Normalize128<<<1, 1>>>(d_ran, 1);

  cudaMemcpy(h_ran, d_ran, sizeof(float)*128, cudaMemcpyDeviceToHost);
  printf("0: %f \n", h_ran[0]);
  
  // free memory
  free(h_ran);
  cudaFree(d_ran);
  return 0;
}