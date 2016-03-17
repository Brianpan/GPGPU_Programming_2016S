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

//segment tree find
__device__ int combineNode(int leftNode, int rightNode){
   if(leftNode == 0 || rightNode == 0){
       return 0;	
   }else{
   	   return (leftNode+rightNode);
   }	
}
__device__ int segment_tree_search(int tidx, int *segment_tree, int left, int right, int lf, int rh){
    if(left == lf && right ==rh){
        return segment_tree[tidx];
    }
    int mid = (left+right)/2;
    int leftChild = tidx*2, rightChild = leftChild+1;

    //include case
    if(lf > mid){
    	return segment_tree_search(rightChild, segment_tree, mid+1, right, lf, rh);
    }else if (rh <= mid){
    	return segment_tree_search(leftChild, segment_tree, left, mid, lf, rh);
    }

    int leftNode = segment_tree_search(leftChild, segment_tree, left, mid, lf, mid);
    int rightNode = segment_tree_search(rightChild, segment_tree, mid+1, right, mid+1, rh);
    //combine two
    return combineNode(leftNode, rightNode);
}

//count global
__global__ void d_countPosition(int *pos, int *segment_tree, int text_size, int seg_tree_size){
    long long int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //out of bound
    if(idx >= text_size){return;}
    
    long long int leaf_shifted = seg_tree_size/2;
    //
    if(segment_tree[leaf_shifted+idx] == 0){
    	pos[idx] = 0;
    }else{
    	//naive n*k
    	int word_posi = 1;
    	// long long int countdown_pivot = idx - 1;
     //    while(countdown_pivot >=0 && segment_tree[leaf_shifted+countdown_pivot] != 0){
     //        word_posi += 1;
     //        countdown_pivot -= 1;
     //    }
     //    pos[idx] = word_posi;
        //segment tree approach n*(log k)
        int max_length = 512;
        int s = 1;
        int base = 0;
        int countdown_pivot = idx - (s+base);
        while((s+base) <= max_length && countdown_pivot >=0){
            int search_result = segment_tree_search(1, segment_tree, 0, seg_tree_size/2-1, countdown_pivot, idx-s-base);
            if(search_result == 0){
                base += s/2;
                s = 1; 
            }else{
                s *= 2;
                if(idx-(s+base) < 0){
                    base += s/2;
                    s = 1;
                }
            }
            countdown_pivot = idx - (s+base);

        }
        pos[idx] = base+1;
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
    
    int blk_size = 32; 
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
       //printf("pos shift: %d ; sib size: %d \n", pos_shifted, to_build_siblings_size);

       pos_shifted = pos_shifted/2;
       to_build_siblings_size = pos_shifted;
       //sync device
       cudaDeviceSynchronize();
       //break; 
    }

    //int h_segment_tree[seg_tree_size];

    // cudaMemcpy(h_segment_tree, d_segment_tree, 
    // 	       seg_tree_size*sizeof(int),
    // 	       cudaMemcpyDeviceToHost);
     
    // for(long long int i = seg_tree_size/2; i< (seg_tree_size/2+600); i=i+2){
    //     printf("tree node %d  %d | nei : %d | parent node: %d | grandparent: %d \n",
    //      i, h_segment_tree[i], h_segment_tree[i+1], h_segment_tree[i/2], h_segment_tree[i/4]);
    // }

    // printf("seg tree size %d \n", seg_tree_size);
    // printf("text size %d\n", text_size);
    
    //count position
    int grid_size = CeilDiv(text_size, blk_size);
    dim3 BLK_SIZE(blk_size, 1, 1);
    dim3 GRID_SIZE(grid_size, 1, 1);

    d_countPosition<<<GRID_SIZE, BLK_SIZE>>>(pos, d_segment_tree, text_size, seg_tree_size);
    
    int *posi_ptr = (int *) malloc(sizeof(int)*text_size);
    cudaMemcpy(posi_ptr, pos, sizeof(int)*text_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i<500;i++){
    	printf("%d,", posi_ptr[i]);
    }
    //free memory
    free(posi_ptr);
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
