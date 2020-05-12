#pragma comment (lib, "cublas.lib")

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cublas.h>
#include <cublas_v2.h>

using namespace std;

const int N = 1<<24;
#define V 0.2
#define T 2

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("")

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
	float *FunctionData = new float[N];
	
	float *host_x;
	float *host_y;
	float *dev_x;
	float *dev_y;

	cudaEvent_t start, stop;

	thrust::host_vector<float> cpumem1(N);
	thrust::host_vector<float> cpumem2(N);

	float time;
        float alpha = V * T;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_x, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_y, N * sizeof(float)));
	
	CUDA_CHECK_RETURN(cudaMallocHost((void **)&host_x, N * sizeof(float)));	
	CUDA_CHECK_RETURN(cudaMallocHost((void **)&host_y, N * sizeof(float)));

	cublasHandle_t cublas_handle;
	CUDA_CHECK_RETURN(cublasCreate(&cublas_handle)); 

	for (int i = 0; i < N; i++) 
        {
	    FunctionData[i] = rand() % 1000;
	    cpumem1[i] = FunctionData[i];
	    cpumem2[i] = FunctionData[i];
            host_x[i] = FunctionData[i];
	    host_y[i] = FunctionData[i];
	}
        cout << endl;

	thrust::device_vector<float> gpumem1 = cpumem1;
	thrust::device_vector<float> gpumem2 = cpumem2;

        CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	saxpy(V * T, gpumem2, gpumem1);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

	printf("Thrust: %f ms\n", time);
	/*for (int i = 0; i < N; i++)
        {
	    cout << gpumem1[i] << " ";
        }
	cout << endl;*/

	CUDA_CHECK_RETURN(cublasSetVector(N, sizeof(float), host_x, 1, dev_x, 1));
	CUDA_CHECK_RETURN(cublasSetVector(N, sizeof(float), host_y, 1, dev_y, 1));

	CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	CUDA_CHECK_RETURN(cublasSaxpy(cublas_handle, N, &alpha, dev_x, 1, dev_y, 1));

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

	CUDA_CHECK_RETURN(cublasGetVector(N, sizeof(float), dev_x, 1, host_x, 1));
	CUDA_CHECK_RETURN(cublasGetVector(N, sizeof(float), dev_y, 1, host_y, 1));

	printf("cuBLAS: %f ms\n", time);
	/*for (int i = 0; i < N; i++)
        {
	    cout << host_y[i] << " ";
	}*/
       
	CUDA_CHECK_RETURN(cudaFreeHost(host_x));
	CUDA_CHECK_RETURN(cudaFreeHost(host_y));
	CUDA_CHECK_RETURN(cudaFree(dev_x));
	CUDA_CHECK_RETURN(cudaFree(dev_y));
	CUDA_CHECK_RETURN(cublasDestroy(cublas_handle));
	
	return 0;
}