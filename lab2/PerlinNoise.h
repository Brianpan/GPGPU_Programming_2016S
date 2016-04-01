#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER 
#endif

#ifndef PERLINNOISE_H

#define PERLINNOISE_H

class PerlinNoise{
	int *p;

	private:
		CUDA_CALLABLE_MEMBER double fade(double t);
		CUDA_CALLABLE_MEMBER double lerp(double t, double a, double b);
		CUDA_CALLABLE_MEMBER double grad(int hash, double x, double y, double z);
	public:
		CUDA_CALLABLE_MEMBER PerlinNoise(int *p);

		CUDA_CALLABLE_MEMBER PerlinNoise(unsigned int seed);

		CUDA_CALLABLE_MEMBER double noise(double x, double y, double z);
};

#endif
