#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

__global__ void ToUpperCaseTransform(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < fsize and input_gpu[idx] != '\n' and input_gpu[idx]!=' ') {
		int ansii_num = input_gpu[idx];
		if(ansii_num >=97 && ansii_num <= 122){
			ansii_num -= 32;
			input_gpu[idx]= (char) ansii_num;
		}
		
	}
}

__device__ bool isUpperCase(const char *input_gpu, int fsize, int idx){
	if(idx < fsize and input_gpu[idx] >=65 && input_gpu[idx] <= 90){
		return true;
	}else{
        return 0;
	}
}
__device__ bool isAlphabet(const char *input_gpu, int fsize, int idx){
    if(idx < fsize and ((input_gpu[idx] >=97 && input_gpu[idx] <= 122) or (input_gpu[idx] >=65 && input_gpu[idx] <= 90))){
        return true;
    }else{
    	return 0;
    }
}

__device__ char toUpperCase(char chr){
    int chr_num = chr;
    if(chr_num >=97 && chr_num <= 122){
    	chr_num -= 32;
    	chr = (char) chr_num;
    }
    return chr;
}

__device__ char toLowerCase(char chr){
	int chr_num = chr;
	if(chr_num >= 65 && chr_num <= 90){
		chr_num += 32;
		chr = (char) chr_num;
	}
	return chr;
}
__global__ void SwithPairCharTransform(char *input_gpu, int fsize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(isAlphabet(input_gpu, fsize, idx)){
        int shifted_posi = 0;
        while(isAlphabet(input_gpu, fsize, idx-shifted_posi)){
            shifted_posi += 1;
        }

        if(shifted_posi %2 == 1 and isAlphabet(input_gpu, fsize, idx+1)){
        	char tmp, tmp_idx_1;
        	if(shifted_posi == 1 and isUpperCase(input_gpu, fsize, idx)){
                tmp = toLowerCase(input_gpu[idx]);
                tmp_idx_1 = toUpperCase(input_gpu[idx+1]); 
        	}else{
        	    tmp = input_gpu[idx];
        	    tmp_idx_1 = input_gpu[idx+1];
        	}
        	__syncthreads();
        	input_gpu[idx] = tmp_idx_1;
        	input_gpu[idx+1] = tmp;
        }
    }  
}

int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (not fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// TODO: do your transform here
	char *input_gpu = text_smem.get_gpu_rw();
	// An example: transform the first 64 characters to '!'
	// Don't transform over the tail
	// And don't transform the line breaks
 	const int blkSize = 512; 
	const int gdSize = (fsize/512)+1;
	const dim3 blockSize(blkSize, 1, 1);
	const dim3 gridSize(gdSize, 1, 1);
	//ToUpperCaseTransform<<<gridSize, blockSize>>>(input_gpu, fsize);
    SwithPairCharTransform<<<gridSize, blockSize>>>(input_gpu, fsize);
	puts(text_smem.get_cpu_ro());
	return 0;
}
