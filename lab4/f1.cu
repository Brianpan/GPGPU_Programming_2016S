__global__ void f1d3(float3 * __restrict__ ptr) {
   float3 v = ptr[threadIdx.x];
   v.x += 1;
   v.y += 1;
   v.z += 1;
   ptr[threadIdx.x] = v;
   return;
}

// __global__ void f1(float4 * __restrict__ ptr) {
//    float4 v = ptr[threadIdx.x];
//    v.x += 1;
//    v.y += 1;
//    v.z += 1;
//    v.w += 1;
//    ptr[threadIdx.x] = v;
//    return;
// }

int main()
{

	// float4 *f1_ptr;
	// cudaMalloc(&f1_ptr, sizeof(float)*128);
	// cudaMemset(f1_ptr, 0, sizeof(float)*128);
	
	float3 *f1_ptr;
	cudaMalloc(&f1_ptr, sizeof(float)*96);
	cudaMemset(f1_ptr, 0, sizeof(float)*96);

	// global fuc
	// Timer t1, t2;
	// t1.Start();
	// f1<<<1,32>>>(f1_ptr);
	f1d3<<<1,32>>>(f1_ptr);
	// t1.Pause();

	// printf_timer(t1);

	cudaFree(f1_ptr);
	return 0;
}
