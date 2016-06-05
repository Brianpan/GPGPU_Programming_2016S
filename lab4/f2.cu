
__global__ void f2d3(float * __restrict__ ptr1, float * __restrict__ ptr2, float * __restrict__ ptr3) {
   ptr1[threadIdx.x] += 1;
   ptr2[threadIdx.x] += 1;
   ptr3[threadIdx.x] += 1;
   return;
}

// __global__ void f2(float * __restrict__ ptr1, float * __restrict__ ptr2, float * __restrict__ ptr3, float * __restrict__ ptr4) {
//    ptr1[threadIdx.x] += 1;
//    ptr2[threadIdx.x] += 1;
//    ptr3[threadIdx.x] += 1;
//    ptr4[threadIdx.x] += 1;
//    return;
// }

int main(int argc, char **argv)
{
	float *f2_ptr;
	// cudaMalloc(&f2_ptr, sizeof(float)*128);
	// cudaMemset(f2_ptr, 0, sizeof(float)*128);
	cudaMalloc(&f2_ptr, sizeof(float)*96);
	cudaMemset(f2_ptr, 0, sizeof(float)*96);

	f2d3<<<1,32>>>(f2_ptr, f2_ptr+32, f2_ptr+64);		
	// f2<<<1,32>>>(f2_ptr, f2_ptr+32, f2_ptr+64, f2_ptr+96);	
	cudaFree(f2_ptr);
	
	return 0;
}
