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
    long long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    long long int tree_idx = shifted_pivot + idx;
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
    	long long int left_tree_node = 2*tree_idx;
    	long long int right_tree_node = left_tree_node+1;

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
    
//count global
__global__ void d_countPosition(int *pos, int *segment_tree, int text_size, int seg_tree_size){
    long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //out of bound
    if(idx >= text_size){return;}
    
    long int leaf_shifted = seg_tree_size>>1;
    //zero condition 
    if(segment_tree[leaf_shifted+idx] == 0){
    	pos[idx] = 0;
	    return;
    }else{
    	//naive n*k
    	// int word_posi = 1;
    	// long long int countdown_pivot = idx - 1;
     //    while(countdown_pivot >=0 && segment_tree[leaf_shifted+countdown_pivot] != 0){
     //        word_posi += 1;
     //        countdown_pivot -= 1;
     //    }
     //    pos[idx] = word_posi;
        //segment tree approach n*(log k)
        //check node is even or odd
        //even start node should move to prev odd
	    int length = 1;
	    int backtrace_id = idx; 
    	if(backtrace_id %2!= 0){
		    backtrace_id -= 1;
		    if(segment_tree[leaf_shifted + backtrace_id] == 0){
			    pos[idx] = length;
			    return; 	
		    }else{
			    length += 1;
		    }
	    }
        //start up trace
    	int max_up_trace = 512;
    	int loop_iv = 2;
    	long int check_idx  = (leaf_shifted + backtrace_id)/2;
    	leaf_shifted >>= 1;
    	do{
    		if(check_idx % 2!= 0){
    			if( segment_tree[check_idx -1]>=loop_iv){
    				length += loop_iv;
    			}else{
    				break;
    			} 	
    		}else if(check_idx %2 == 0 && check_idx == leaf_shifted){
    			break;
    		}

    		check_idx >>= 1;
    		loop_iv <<= 1;
    		leaf_shifted >>= 1;
    	}while(loop_iv <= max_up_trace);
        
        //down trace if check_idx = 0
    	if(segment_tree[check_idx/2] == 0 && !(check_idx == leaf_shifted && segment_tree[check_idx] > 0)){
    		//move down one sibling
    		check_idx -= 1;
    		//start trace
    		long int left_node;
    		long int right_node;

    		if(segment_tree[check_idx] == 0){
    			while(check_idx < seg_tree_size/2){
    				left_node = check_idx << 1;
    				right_node = left_node + 1;
    				if(segment_tree[right_node] > 0){
    					length += segment_tree[right_node];
    					check_idx <<= 1; 
    				}else{
    					check_idx = check_idx*2 + 1;
    				}
    			}
    		}
    	}	
    	    pos[idx] = length;
    }
    return;

}

//cpu part
void CountPosition(const char *text, int *pos, int text_size)
{
    long long int seg_tree_size = SegmentTreeSize(text_size); 
    long long int pos_shifted = seg_tree_size/2; 
    long long int to_build_siblings_size = pos_shifted;
    int *d_segment_tree;
    cudaMalloc(&d_segment_tree, seg_tree_size*sizeof(int));
    
    int blk_size = 256; 
    while(pos_shifted > 0){
       //do __global__ set segment tree
       long long int grid_size = CeilDiv(to_build_siblings_size, blk_size);
       dim3 BLK_SIZE(blk_size, 1, 1);
       dim3 GRID_SIZE(grid_size, 1, 1);

       if(pos_shifted == seg_tree_size/2){
           buildSegTree<<<GRID_SIZE, BLK_SIZE>>>(pos_shifted, d_segment_tree, text, text_size);       
       }else{
           buildSegTree<<<GRID_SIZE, BLK_SIZE>>>(pos_shifted, d_segment_tree);	
       }
       //update to parent for constructing parents

       pos_shifted = pos_shifted/2;
       to_build_siblings_size = pos_shifted;
       //sync device
       cudaDeviceSynchronize();
    }
    
    //count position
    int grid_size = CeilDiv(text_size, blk_size);
    dim3 BLK_SIZE(blk_size, 1, 1);
    dim3 GRID_SIZE(grid_size, 1, 1);

    d_countPosition<<<GRID_SIZE, BLK_SIZE>>>(pos, d_segment_tree, text_size, seg_tree_size);

    //free memory
    cudaFree(d_segment_tree);
    return;
}

struct filter_trans{
    __host__ __device__ bool operator()(const int &pos){
        return pos == 1;    
    }
};

struct is_head_trans{
    __host__ __device__ int operator()(const int &pos, const int &is_head){
        return (pos*is_head - 1);
    }
};

struct remove_minus_one_trans{
    __host__ __device__ bool operator()(const int &pos){
        return(pos >= 0);
    }
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	//use thrust pointer to manipulate thrust algorithms
    thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO
    // if 1 is 1 otherwise are 0
    thrust::transform(pos_d, pos_d+text_size, flag_d,filter_trans());
    //calculate count
    nhead = thrust::count(flag_d, flag_d+text_size, 1);
    
    thrust::sequence(flag_d+text_size, flag_d+2*text_size, 1);
    // multiply minus 1 is larger than 0 is answer
    thrust::transform(flag_d+text_size, flag_d+2*text_size, flag_d, flag_d, is_head_trans());
	
    // copy to head_d
    // manipulate the address in memory directly
    thrust::copy_if(flag_d, flag_d+text_size, head_d, remove_minus_one_trans());

    cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
