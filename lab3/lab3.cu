#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
  const float *background,
  const float *target,
  const float *mask,
  float *output,
  const int wb, const int hb, const int wt, const int ht,
  const int oy, const int ox
)
{
  const int yt = blockIdx.y * blockDim.y + threadIdx.y;
  const int xt = blockIdx.x * blockDim.x + threadIdx.x;
  const int curt = wt*yt+xt;
  if (yt < ht and xt < wt and mask[curt] > 127.0f) {
    const int yb = oy+yt, xb = ox+xt;
    const int curb = wb*yb+xb;
    if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
      output[curb*3+0] = target[curt*3+0];
      output[curb*3+1] = target[curt*3+1];
      output[curb*3+2] = target[curt*3+2];
    }
  }
}

//right hand side
__global__ void CalculateFixed(
  const float *background,
  const float *target,
  const float *mask,
  float *fixed,
  int *neigh_t,
  const int wb, const int hb, const int wt, const int ht,
  const int oy, const int ox
){
  const int yt = blockIdx.y * blockDim.y + threadIdx.y;
  const int xt = blockIdx.x * blockDim.x + threadIdx.x;
  const int curt = wt*yt + xt;
  
  if(yt >= ht or xt >= wt or mask[curt] <= 127.0f){ 
    return;
  }
  //is in the bound
 
  const int yb = oy + yt, xb = ox + xt;
  
  // add offset to background
  if(0 <= yb and yb < hb and 0 <= xb and xb < wb){
    // margin is background
    int d_move[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    neigh_t[curt] = 4;

    fixed[curt*3 + 0] = 4*target[curt*3 + 0];
    fixed[curt*3 + 1] = 4*target[curt*3 + 1];
    fixed[curt*3 + 2] = 4*target[curt*3 + 2];

    for(int neighbor = 0; neighbor < 4; neighbor++){
      int x_tmp, y_tmp, t_x_tmp, t_y_tmp;
      x_tmp = xb + d_move[neighbor][0];
      y_tmp = yb + d_move[neighbor][1];
      
      t_x_tmp = xt + d_move[neighbor][0];
      t_y_tmp = yt + d_move[neighbor][1];

      int n_curt = wb*y_tmp + x_tmp;
      int t_curt = t_y_tmp*wt + t_x_tmp;
      
      if(t_x_tmp >= 0 and t_x_tmp < wt and t_y_tmp >= 0 and t_y_tmp < ht){
        // add number of neighbor
        

        // 3 dim target
        fixed[curt*3 + 0] -= target[t_curt*3 + 0];
        fixed[curt*3 + 1] -= target[t_curt*3 + 1];
        fixed[curt*3 + 2] -= target[t_curt*3 + 2];
        
      }else{
        // 3 dim target
        fixed[curt*3 + 0] -=  target[curt*3 + 0];
        fixed[curt*3 + 1] -=  target[curt*3 + 1];
        fixed[curt*3 + 2] -=  target[curt*3 + 2];
      }

      if(x_tmp >= 0 and x_tmp < wb and y_tmp >=0 and y_tmp < hb and ((t_x_tmp < 0 or t_x_tmp >= wt or t_y_tmp < 0 or t_y_tmp >= ht) or mask[t_curt] <= 127.0f)){
        for(int i = 0; i < 3; i++){
          fixed[curt*3 + i] += background[n_curt*3 + i];
        }
      }

      // less than 4 point if out of bound with background image
      if(x_tmp < 0 or y_tmp < 0 or x_tmp >= wb or y_tmp >= hb){
        neigh_t[curt] -= 1;
      } 
    }
  }

  return;
}

__global__ void PoissonImageCloningIteration(
  const float *fixed, 
  const float *mask, 
  const int *neigh_t,
  float *updated,
  float *new_data, 
  const int wt, 
  const int ht
){
  const int yt = blockIdx.y*blockDim.y + threadIdx.y;
  const int xt = blockIdx.x*blockDim.x + threadIdx.x;
  const int curt = yt*wt + xt;
  
  if(xt >= wt || yt >= ht || mask[curt] <= 127.0f){
    return;
  }

  // check neighbor to form equation
  // sparse matrix
  //right hand value
  float temp[3];
  temp[0] = fixed[curt*3];
  temp[1] = fixed[curt*3 + 1];
  temp[2] = fixed[curt*3 + 2];
  
  // do b-AX
  int d_move[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  for(int i = 0; i < 4; i++){
    int x_tmp, y_tmp, t_cur;

    // x axis
    x_tmp = xt + d_move[i][0];
    y_tmp = yt + d_move[i][1]; 
    t_cur = x_tmp + y_tmp*wt;

    // if is inside
    if(x_tmp >= 0 and x_tmp < wt and y_tmp >= 0 and y_tmp < ht and mask[t_cur] > 127.0f){
      for(int i = 0 ; i < 3 ; i ++){
        temp[i] += updated[3*t_cur + i];
      }
    } 
  }

  for(int i = 0 ; i < 3 ; i ++){
    // minus n*current point
    temp[i] /= neigh_t[curt];
    
    // save to buf
    new_data[3*curt + i] = temp[i];
  }

  return;
}

void PoissonImageCloning(
  const float *background,
  const float *target,
  const float *mask,
  float *output,
  const int wb, const int hb, const int wt, const int ht,
  const int oy, const int ox
)
{ 
  // poisson image editing
  float *fixed, *buf1, *buf2;
  int *neigh_t;

  cudaMalloc(&neigh_t, sizeof(int)*wt*ht);
  cudaMalloc(&fixed, 3*sizeof(float)*wt*ht);
  cudaMalloc(&buf1, 3*sizeof(float)*wt*ht);
  cudaMalloc(&buf2, 3*sizeof(float)*wt*ht);

  // initialize neighbor to 0
  cudaMemset(neigh_t, 0, sizeof(int)*wt*ht);
  cudaMemset(fixed, 0, 3*sizeof(float)*wt*ht);
  
  
  dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
  // right hand side
  CalculateFixed<<<gdim, bdim>>>(background, target, mask, fixed, neigh_t,
                 wb, hb, wt, ht, oy, ox);
  cudaDeviceSynchronize();

  // poisson initialized x = 0
  // cudaMemset(buf1, 0, 3*sizeof(float)*wt*ht);
  cudaMemset(buf2, 0, 3*sizeof(float)*wt*ht);
  cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

  // iter
  for(int i= 0 ; i < 20000; i++){
    PoissonImageCloningIteration<<<gdim, bdim>>>(
      fixed, mask, neigh_t, buf1 ,buf2, wt, ht
    );
    cudaDeviceSynchronize();
    cudaMemcpy(buf1, buf2, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
  }

  //easy post
  cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
  SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
    background, buf1, mask, output,
    wb, hb, wt, ht, oy, ox
  );

  cudaFree(neigh_t);
  cudaFree(fixed);
  cudaFree(buf1);
  cudaFree(buf2);
}

