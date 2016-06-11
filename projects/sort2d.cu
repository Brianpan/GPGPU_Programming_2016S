#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <iostream>
#include <stdlib.h>
#define VIEWERS 5
#define NSORTS 2
#define DSIZE 10

int my_mod_start = 0;
int my_mod(){
  return (my_mod_start++)/DSIZE;
}

bool validate(thrust::device_vector<float> &d1, thrust::device_vector<float> &d2){
  return thrust::equal(d1.begin(), d1.end(), d2.begin());
}


struct sort_functor
{
  thrust::device_ptr<int> data;
  int dsize;
  __host__ __device__
  void operator()(int start_idx)
  {
    thrust::sort(thrust::device, data+(dsize*start_idx), data+(dsize*(start_idx+1)));
  }
};



#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}
float rand_self(){
  return (float) rand()/RAND_MAX;
}
int main(){
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, (16*DSIZE*NSORTS));
  thrust::host_vector<float> h_data(DSIZE*NSORTS);
  thrust::generate(h_data.begin(), h_data.end(), rand_self);
  thrust::device_vector<float> d_data = h_data;

  // first time a loop
  thrust::device_vector<float> d_result1 = d_data;
  thrust::device_ptr<float> r1ptr = thrust::device_pointer_cast<float>(d_result1.data());
  unsigned long long mytime = dtime_usec(0);
  for (int i = 0; i < NSORTS; i++)
    thrust::sort(r1ptr+(i*DSIZE), r1ptr+((i+1)*DSIZE));
  cudaDeviceSynchronize();
  mytime = dtime_usec(mytime);
  std::cout << "loop time: " << mytime/(float)USECPSEC << "s" << std::endl;

  //vectorized sort
  thrust::device_vector<float> d_result2 = d_data;
  thrust::host_vector<int> h_segments(DSIZE*NSORTS);
  thrust::generate(h_segments.begin(), h_segments.end(), my_mod);
  thrust::device_vector<int> d_segments = h_segments;
  mytime = dtime_usec(0);
  thrust::stable_sort_by_key(d_result2.begin(), d_result2.end(), d_segments.begin());
  
  thrust:: host_vector<int> h_rank = d_segments;
  thrust:: host_vector<float> h_dd = d_result2;
  for(int i = 0; i < DSIZE*NSORTS; i ++){
    if(i%DSIZE==0){
      printf("------\n");
    }
    printf("---key: %f rank: %d ---\n", h_dd[i], h_rank[i]);
  }
  cudaDeviceSynchronize();
  thrust::stable_sort_by_key(d_segments.begin(), d_segments.end(), d_result2.begin());

  cudaDeviceSynchronize();

  

  float *hd_data= (float *) malloc(sizeof(float)*DSIZE*NSORTS);
  float *raw_ptr = thrust::raw_pointer_cast(d_result2.data());

  cudaMemcpy(hd_data, raw_ptr, sizeof(float)*DSIZE*NSORTS, cudaMemcpyDeviceToHost);
  for(int i = 0; i < 2*DSIZE;i++){
    if(i%DSIZE==0){
      printf("---------\n");
    }
    printf("data: %f sorted: %f\n", h_data[i], hd_data[i]);
  }
  mytime = dtime_usec(mytime);
  std::cout << "vectorized time: " << mytime/(float)USECPSEC << "s" << std::endl;
  if (!validate(d_result1, d_result2)) std::cout << "mismatch 1!" << std::endl;
  // //nested sort
  // thrust::device_vector<int> d_result3 = d_data;
  // sort_functor f = {d_result3.data(), DSIZE};
  // thrust::device_vector<int> idxs(NSORTS);
  // thrust::sequence(idxs.begin(), idxs.end());
  // mytime = dtime_usec(0);
  // thrust::for_each(idxs.begin(), idxs.end(), f);
  // cudaDeviceSynchronize();
  // mytime = dtime_usec(mytime);
  // std::cout << "nested time: " << mytime/(float)USECPSEC << "s" << std::endl;
  // if (!validate(d_result1, d_result3)) std::cout << "mismatch 2!" << std::endl;
  return 0;
}