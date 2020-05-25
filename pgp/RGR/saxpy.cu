#pragma comment (lib, "cublas.lib")

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cublas.h>
#include <cublas_v2.h>

using namespace std;

const int N = 1 << 26;
#define V 0.2
#define T 2

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("")


__global__ void saxpyblas(float *x, float *y, float *z) 
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	z[i] = x[i] * (V * T) + y[i];
}

struct functor 
{
    const float koef;
    functor(float _koef) : koef(_koef) {}
    __host__ __device__ float operator()(float x, float y) 
    { 
        return koef * x + y; 
    }
};

void saxpy(float _koef, thrust::device_vector<float> &x, thrust::device_vector<float> &y)
{
    functor func(_koef);
    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}

int main()
{
	srand(time(NULL));
	
	cudaEvent_t start, stop;

	float time;
        float alpha = V * T;	
	float max[27];
	int k = 0;
	for(int i = 0; i < 27; i++)
	{
		max[i] = 0;
	}

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	cublasHandle_t cublas_handle;
	CUDA_CHECK_RETURN(cublasCreate(&cublas_handle)); 

    	cudaDeviceProp deviceProp;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, 0));

	int blocks = 0;
	int threads = 0;
	int count = 1;

	for(int i = 0; i < 10; i++)
	{
		count = 1;
		while(count <= N)
		{
			float *x;
			float *y;
			float *z;

			float *x1;
			float *y1;
			float *z1;

			CUDA_CHECK_RETURN(cudaMalloc((void **)&x1, count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&y1, count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&z1, count * sizeof(float)));

			CUDA_CHECK_RETURN(cudaMallocHost((void **)&x, count * sizeof(float)));	
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&y, count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&z, count * sizeof(float)));

			for(int i = 0; i < count; i++)
			{
				x[i] = rand() % 1000;
				y[i] = rand() % 1000;
			}

    			CUDA_CHECK_RETURN(cudaMemcpy(x1, x, count * sizeof(float), cudaMemcpyHostToDevice));
    			CUDA_CHECK_RETURN(cudaMemcpy(y1, y, count * sizeof(float), cudaMemcpyHostToDevice));
	
			if(count <= deviceProp.maxThreadsPerBlock)
			{
				threads = count;
				blocks = 1;	
			}
			else
			{
				threads = deviceProp.maxThreadsPerBlock;
				blocks = count / 1024;
			}

       			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
			saxpyblas <<< blocks, threads >>> (x1, y1, z1); 	

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
	
    			CUDA_CHECK_RETURN(cudaMemcpy(z, z1, count * sizeof(float), cudaMemcpyDeviceToHost));

			printf("%d - CUDA: %f ms\n", count, time);
			max[k] += time;

			CUDA_CHECK_RETURN(cudaFreeHost(x));
			CUDA_CHECK_RETURN(cudaFreeHost(y));
			CUDA_CHECK_RETURN(cudaFreeHost(z));

			CUDA_CHECK_RETURN(cudaFree(x1));
			CUDA_CHECK_RETURN(cudaFree(y1));
			CUDA_CHECK_RETURN(cudaFree(z1));

			count <<= 1;
			k++;
		}
		k = 0;
	}

	for(int i = 0; i < 27; i++)
	{
		cout << max[i] / 10 * 1000 << endl;
		max[i] = 0;
	}

	for(int i = 0; i < 10; i++)
	{
		count = 1;
		while(count <= N)
		{	
			thrust::host_vector<float> cpumem1(count);
			thrust::host_vector<float> cpumem2(count);

			for (int i = 0; i < count; i++) 
      	  		{
	   			cpumem1[i] = rand() % 1000;
	    			cpumem2[i] = rand() % 1000;
			}

			thrust::device_vector<float> gpumem1 = cpumem1;
			thrust::device_vector<float> gpumem2 = cpumem2;

       			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

			saxpy(V * T, gpumem2, gpumem1);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

			printf("%d - Thrust: %f ms\n", count, time);
			max[k] += time;

			count <<= 1;
			k++;
		}
		k = 0;
	}

	for(int i = 0; i < 27; i++)
	{
		cout << max[i] / 10 * 1000 << endl;
		max[i] = 0;
	}

	for(int i = 0; i < 10; i++)
	{
		count = 1;
		while(count <= N)
		{	
			float *host_x;
			float *host_y;

			float *dev_x;
			float *dev_y;

			CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_x, count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_y, count * sizeof(float)));
	
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&host_x, count * sizeof(float)));	
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&host_y, count * sizeof(float)));	

			for (int i = 0; i < count; i++) 
       			{
            			host_x[i] = rand() % 1000;
	    			host_y[i] = rand() % 1000;
			}

			CUDA_CHECK_RETURN(cublasSetVector(count, sizeof(float), host_x, 1, dev_x, 1));
			CUDA_CHECK_RETURN(cublasSetVector(count, sizeof(float), host_y, 1, dev_y, 1));

			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

			CUDA_CHECK_RETURN(cublasSaxpy(cublas_handle, count, &alpha, dev_x, 1, dev_y, 1));

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

			CUDA_CHECK_RETURN(cublasGetVector(count, sizeof(float), dev_x, 1, host_x, 1));
			CUDA_CHECK_RETURN(cublasGetVector(count, sizeof(float), dev_y, 1, host_y, 1));

			printf("%d - cuBLAS: %f ms\n", count, time);
			max[k] += time;

			CUDA_CHECK_RETURN(cudaFreeHost(host_x));
			CUDA_CHECK_RETURN(cudaFreeHost(host_y));

			CUDA_CHECK_RETURN(cudaFree(dev_x));
			CUDA_CHECK_RETURN(cudaFree(dev_y));

			count <<= 1;
			k++;
		}
		k = 0;
        }

	for(int i = 0; i < 27; i++)
	{
		cout << max[i] / 10 * 1000 << endl;
		max[i] = 0;
	}

	CUDA_CHECK_RETURN(cublasDestroy(cublas_handle));
	
	return 0;
}