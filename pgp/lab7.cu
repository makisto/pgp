#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <stdio.h>


using namespace std;

#define N 10
#define V 0.2
#define T 2
struct functor {
	const float koef;
	functor(float _koef) : koef(_koef) {}
	__host__ __device__ float operator()(float x, float y) { return y + koef * (x - y); }
};
void iteration(float _koef, thrust::device_vector<float> &x,
	thrust::device_vector<float> &y)
{
	functor func(_koef);
	thrust::transform(x.begin(),x.end(),y.begin(),y.begin(),func);
		
}

__global__ void kernel(float *f, float *res) {
	int cur = threadIdx.x + blockDim.x * blockIdx.x;
	int prev = cur - 1;
	if (prev == -1)
		prev = N - 1;
	if (cur >= 0 && cur < N) {
		res[cur] = f[cur] + (V * T) * (f[prev] - f[cur]);
	}
}

int main()
{
	setlocale(LC_ALL,"Russian");
	float Function[N];
	float FunctionData[N];
	float *frez;
	float *tempa;

	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	for (int i = 0; i < N; i++) {
		FunctionData[i] = rand() % 100;
		Function[i] = FunctionData[i];
	}
	cudaMalloc((void **)&frez, sizeof(float) * N);
	cudaMalloc((void **)&tempa, sizeof(float) * N);

	cudaMemcpy(tempa, Function, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);
	for (int i = 0; i < 1000; i++) {
		kernel << <1, N >> > (tempa, frez);
		cudaMemcpy(Function, frez, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(tempa, frez, sizeof(float) * N, cudaMemcpyHostToDevice);
	}
	
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << std::endl;
	printf("lead time: %f ms\n", time);
    // Add vectors in parallel.
   
	thrust::host_vector<float> cpumem1(N);
	thrust::host_vector<float> cpumem2(N);

	for (int i = 0; i < N; i++)
	{
		cpumem1[i] = FunctionData[i];

		(i - 1 >= 0) ? cpumem2[i] = FunctionData[i - 1] : cpumem2[i] = FunctionData[N - 1];
	}
	thrust::device_vector<float> gpumem1 = cpumem1;
	thrust::device_vector<float> gpumem2 = cpumem2;

	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);

	for (int i = 0; i < 1000; i++)
	{
		iteration(V*T, gpumem2, gpumem1);
		for (int i = 0; i < N; i++)
		{
			cpumem1 = gpumem1;
			gpumem2 = cpumem1;
			(i - 1 >= 0) ? cpumem2[i] = cpumem1[i - 1] : cpumem2[i] = cpumem1[N - 1];
		}
		gpumem1 = cpumem2;
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << std::endl;
	printf("lead time: %f ms\n", time);
	/*for (int i = 0; i < N; i++)
		cout << gpumem1[i] << " ";*/


    return 0;
}

float TransportEquation()
{
	return 0;
}
