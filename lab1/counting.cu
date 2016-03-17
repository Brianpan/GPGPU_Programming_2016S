#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void buildSegTree(int shifted_pivot, int *segment_tree, const char *text=NULL, int text_size=0){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tree_idx = shifted_pivot + idx;
    //leaf
    if(text){
    	int leaf_val = 0;
        if(idx < text_size){
        	//be careful for using single quote
        	if(text[idx] != '\n'){
                leaf_val = 1;
        	}
        }
        segment_tree[tree_idx] = leaf_val;    
    //not leaf
    }else{
    	int left_tree_node = 2*tree_idx;
    	int right_tree_node = left_tree_node+1;

    	if(segment_tree[left_tree_node] == 0 || segment_tree[right_tree_node] == 0){ 
            segment_tree[tree_idx] = 0;
        }else{
        	segment_tree[tree_idx] = segment_tree[left_tree_node] + segment_tree[right_tree_node];
        }    
    }
    return;
}

__host__ int SegmentTreeSize(int text_size){
	int s = 1;
	for(;s<text_size;s<<=1);
	return s<<1;	
}

void CountPosition(const char *text, int *pos, int text_size)
{
    int seg_tree_size = SegmentTreeSize(text_size); 
    int pos_shifted = seg_tree_size/2; 
    int to_build_siblings_size = pos_shifted;
    int *d_segment_tree;
    cudaMalloc(&d_segment_tree, seg_tree_size*sizeof(int));
    
    int blk_size = 32; 
    while(pos_shifted > 0){
       //do __global__ set segment tree
       int grid_size = CeilDiv(to_build_siblings_size, blk_size);
       dim3 BLK_SIZE(blk_size, 1, 1);
       dim3 GRID_SIZE(grid_size, 1, 1);

       if(pos_shifted == seg_tree_size/2){
           buildSegTree<<<GRID_SIZE, BLK_SIZE>>>(pos_shifted, d_segment_tree, text, text_size);       
       }else{
           buildSegTree<<<GRID_SIZE, BLK_SIZE>>>(pos_shifted, d_segment_tree);	
       }
       //update to parent for constructing parents
       printf("pos shift: %d ; sib size: %d \n", pos_shifted, to_build_siblings_size);

       pos_shifted = pos_shifted/2;
       to_build_siblings_size = pos_shifted;
       //sync device
       cudaDeviceSynchronize();
       //break; 
    }

    int h_segment_tree[seg_tree_size];

    cudaMemcpy(h_segment_tree, d_segment_tree, 
    	       seg_tree_size*sizeof(int),
    	       cudaMemcpyDeviceToHost);
     
    // for(long long int i = seg_tree_size/2; i< (seg_tree_size/2+600); i=i+2){
    //     printf("tree node %d  %d | nei : %d | parent node: %d | grandparent: %d \n",
    //      i, h_segment_tree[i], h_segment_tree[i+1], h_segment_tree[i/2], h_segment_tree[i/4]);
    // }

    // printf("seg tree size %d \n", seg_tree_size);
    // printf("text size %d\n", text_size);

    //count position
    cudaFree(d_segment_tree);
    return;
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
