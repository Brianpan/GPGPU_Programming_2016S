#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <random>

#include "SyncedMemory.h"
#include "lab2.h"
#include "PerlinNoise.h"



static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }

//function array assign
std::vector<int> assign_p(){
	std::vector<int> p;
	p = {
		151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
		8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
		35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,
		134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
		55,46,245,40,244,102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
		18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
		250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
		189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 
		43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,
		97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,
		107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };

	p.insert(p.end(), p.begin(), p.end());
	return p;	
}

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

__device__ int3 RGBtoHSL(int3 rgb){
	int y = 0.299*rgb.x + 0.587*rgb.y + 0.113*rgb.z;
	int u = -0.169*rgb.x - 0.331*rgb.y + 0.5*rgb.z + 128;
	int v = 0.5*rgb.x - 0.419*rgb.y - 0.081*rgb.z + 128;

	int3 yuv = make_int3(y, u, v);

	return yuv;
}


__device__ double turbance(double x, double y, double z, int octave, double persistence, PerlinNoise pn){
	double total = 0;
	double freq = 1;
	double amp = 1;
	double maxVal = 0;

	for(int i = 0; i < octave ; i ++){
		total += fabs(pn.noise(x*freq, y*freq, z*freq))*amp;
		
		maxVal += amp; 
		amp *= persistence;
		freq *= 2.0;
	}
	
	return total/maxVal;
}

__device__ double gLine(int x, int y, int w, int h){
	int _x = abs(x-w/2);
	int _y = abs(y-h/2);
	int l = _x*_x+_y*_y;
	if(sqrtf(l) > 200){
		if(y>x){
			return y/x;
		}
		return x/y;
	}else{
		return l;
	}
}

__device__ double lerp(double a, double b, double t){
	return a + (b-a) * t;
}

__global__ void perlinTransform(int t, int NFRAME, int width, int height, uint8_t *yuv, int *d_p){
	const int2 thread_2d_pos = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
										 blockIdx.y*blockDim.y + threadIdx.y);
	int posX = thread_2d_pos.x;
	int posY = thread_2d_pos.y;

	double PI = 3.1415926;

	PerlinNoise pn(d_p);
	
	if(posX >= width || posY >= height){
		return;
	}	

	double xyPeriod = 12.0; //number of rings
    double turbPower = 0.4; //makes twists
    int octave = 6; //initial size of the turbulence
	double persistence = 0.5;

	double x = (posX - width/2) / double(width);
	double y = (posY - height/2) / double(height);
	
	//do wood like
	double noise = turbance(x, y, 0, octave, persistence, pn);
	double noise2 = turbance(x, y, t/6, octave, persistence, pn);

	double distValue = sqrtf(x*x + y*y) + turbPower*noise;		

	double sinValue = 128.0 *fabs(sinf(2*(t/6)*xyPeriod*distValue*PI));	
	int color_r = 80+sinValue;
	int color_g = floor(255*cosf(noise2+gLine(x, y, width, height)));
	
	double total = pn.noise(x, y, t*2)*20;
	total = total - floor(total);
	int color_b = lerp(30, 120, total);

	int3 hsl = RGBtoHSL(make_int3(color_r, color_g, color_b));
	
	yuv[posY*width+posX] = hsl.x;
	yuv[(posY*width+posX)+width*height] = hsl.y;
	yuv[(posY*width+posX)+2*width*height] = hsl.z;
	
	return;
}


__global__ void sampleTransform(int w, int h, uint8_t *yuv, uint8_t *unsample){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = w*y + x;
	if(x >= w || y >= h){
		return;
	}	
	//color y
	yuv[idx] = unsample[idx];

	if(x & 1 || y & 1){
		return;
	}
	
	//color u
	yuv[x/2+y*w/4+w*h] = (unsample[idx+w*h] + unsample[idx+1+w*h] + unsample[(y+1)*w + x + w*h] + unsample[(y+1)*w + x + 1 + w*h])/4;
	//color v
	yuv[x/2+y*w/4+5*w*h/4] = (unsample[idx+2*w*h] + unsample[idx+1+2*w*h] + unsample[(y+1)*w + x + 2*w*h] + unsample[(y+1)*w + x + 1 + 2*w*h])/4;
}

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	dim3 blockSize(32, 32);
	dim3 gridSize(CeilDiv(W, blockSize.x), CeilDiv(H, blockSize.y));
	std::vector<int> p = assign_p();
	int *h_p_arr = &p[0];
	int *d_p_arr;
	cudaMalloc(&d_p_arr, sizeof(int)*512);
	cudaMemcpy(d_p_arr, h_p_arr, sizeof(int)*512, cudaMemcpyHostToDevice);

	// unsample frame
	unsigned FRAME_SIZE = 3*W*H;
	MemoryBuffer<uint8_t> frameb(FRAME_SIZE);
	auto total_frames = frameb.CreateSync(FRAME_SIZE);

	perlinTransform<<<gridSize, blockSize>>>(impl->t, NFRAME, W, H, total_frames.get_gpu_wo(), d_p_arr);
	//cudaMemset(yuv+W*H, 128, W*H/2);
	sampleTransform<<<gridSize, blockSize>>>(W, H, yuv, total_frames.get_gpu_wo());
	++(impl->t);
	
	//release
	cudaFree(d_p_arr);
}


//Perlin noise Class
__device__ PerlinNoise::PerlinNoise(int *p_arr){
	p = p_arr;	
}

__device__ double PerlinNoise::noise(double x, double y, double z){
	//
	int X = (int) floor(x) & 255;
	int Y = (int) floor(y) & 255;
	int Z = (int) floor(z) & 255;
	
	//trans to float point    
	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	//use for interpolate 
	double u = fade(x);
	double v = fade(y);
	double w = fade(z);

	// eight corners
	double aaa = p[ p[p[X] + Y] + Z];
	double aba = p[ p[p[X] + Y+1] + Z];
	double baa = p[p[p[X+1] + Y] + Z];
	double bba = p[p[p[X+1]+ Y+1] + Z];

	double aab = p[p[p[X] + Y] + Z+1]; 
	double abb = p[p[p[X] + Y+1] + Z+1];
	double bab = p[p[p[X+1] + Y+1] + Z+1];
	double bbb = p[p[p[X+1] + Y+1] + Z+1];

	// 3d interpolate
	double x1, x2, y1, y2, res;

	// z=0 interpolate
	x1 = lerp(grad(aaa, x, y, z),
			  grad(baa, x-1, y, z),
			  u);
	x2 = lerp(grad(aba, x, y-1, z),
			  grad(bba, x-1, y-1, z),
			  u);
	y1 = lerp(x1, x2, v);

	// z=1 interpolate
	x1 = lerp(grad(aab, x, y, z-1),
			  grad(bab, x-1, y, z-1),
			  u);
	x2 = lerp(grad(abb, x, y-1, z-1),
			  grad(bbb, x-1, y-1, z-1),
			  u);
	y2 = lerp(x1, x2, v);

	//z index interpolate
	res = lerp(y1, y2, w);
	
	//rescale to 0-1 
	return res;

}

__device__ double PerlinNoise::fade(double t){
	return t * t * t * (t * (t*6 - 15) + 10);
}

__device__ double PerlinNoise::lerp(double a, double b, double t){
	return a + (b-a) * t;
}

//random pick 12 gradient by hash int
__device__ double PerlinNoise::grad(int hash, double x, double y, double z){
	
	int h = hash & 15;
	double u = h < 8 ? x : y,
		   v = h < 4 ? y : (h == 12 || h == 14 ? x : z);

	//last 2 bits for +/-	   
	return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);	   
}