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

const int N = 1 << 10;
#define V 0.2
#define T 2

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("")

__global__ void sgemmblas(float *a, float *b, float *c, int count) 
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0;
	int ia = count * bx + tx;
	int ib = bx + tx;

	for(int k = 0; k < count; k++)
	{
		sum += a[ia + k] * b[ib + k * count];
	}
	int ic = count * by * bx;
	c[ic + count * ty + tx] = sum;
}

struct dp
{
	float *A, *B;
  	int count;
  	dp(float *_A, float *_B, int _count): A(_A), B(_B), count(_count){};
  	__host__ __device__
  	float operator()(size_t idx)
	{
   		float sum = 0.0f;
    		int row = idx / count;
    		int col = idx - (row * count);
    		for (int i = 0; i < count; i++)
		{
      			sum += A[col + row * i] * B[col + row * i];
		}
    		return sum;
	}
};

void sgemm(int count, thrust::device_vector<float> &data, thrust::device_vector<float> &other, thrust::device_vector<float> &result)
{
	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(count * count), result.begin(), dp(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(other.data()), count));
}

int main()
{
	srand(time(NULL));
	
	cudaEvent_t start, stop;

	float time;
        float alpha = 1;

	float max[11];
	int k = 0;

	for(int i = 0; i < 11; i++)
	{
		max[i] = 0;
	}

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	cublasHandle_t cublas_handle;
	CUDA_CHECK_RETURN(cublasCreate(&cublas_handle)); 

    	cudaDeviceProp deviceProp;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, 0));

	int count = 1;
	for(int i = 0; i < 10; i++)
	{
		count = 1;
		while(count <= N)
		{
			float *a;
			float *b;
			float *c;

			float *a_dev;
			float *b_dev;
			float *c_dev;

			CUDA_CHECK_RETURN(cudaMalloc((void **)&a_dev, count * count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&b_dev, count * count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&c_dev, count * count * sizeof(float)));

			CUDA_CHECK_RETURN(cudaMallocHost((void **)&a, count * count * sizeof(float)));	
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&b, count * count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&c, count * count * sizeof(float)));

			for(int i = 0; i < count; i++)
			{
				for(int j = 0; j < count; j++)
				{
					a[i * count + j] = rand() % 1000;
					b[i * count + j] = rand() % 1000;	
					c[i * count + j] = 0;		
				}
			}

    			CUDA_CHECK_RETURN(cudaMemcpy(a_dev, a, count * count * sizeof(float), cudaMemcpyHostToDevice));
    			CUDA_CHECK_RETURN(cudaMemcpy(b_dev, b, count * count * sizeof(float), cudaMemcpyHostToDevice));
    			CUDA_CHECK_RETURN(cudaMemcpy(c_dev, c, count * count * sizeof(float), cudaMemcpyHostToDevice));

       			dim3 threadBlock(1024, 1);
			dim3 blockGrid(count / 1024 + 1, 1, 1);

       			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
			sgemmblas <<< blockGrid, threadBlock >>> (a_dev, b_dev, c_dev, count); 		

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

    			CUDA_CHECK_RETURN(cudaMemcpy(c, c_dev, count * count * sizeof(float), cudaMemcpyDeviceToHost));
	
			printf("%d - CUDA: %f ms\n", count, time);
			max[k] += time;

			CUDA_CHECK_RETURN(cudaFreeHost(a));
			CUDA_CHECK_RETURN(cudaFreeHost(b));
			CUDA_CHECK_RETURN(cudaFreeHost(c));

			CUDA_CHECK_RETURN(cudaFree(a_dev));
			CUDA_CHECK_RETURN(cudaFree(b_dev));
			CUDA_CHECK_RETURN(cudaFree(c_dev));

			count <<= 1;
			k++;
		}
		k = 0;
	}
	for(int i = 0; i < 11; i++)
	{
		cout << max[i] / 10 << endl;
		max[i] = 0;
	}

	for(int i = 0; i < 10; i++)
	{
		count = 1;
		while(count <= N)
		{	
  			thrust::device_vector<float> data(count * count, 2);
  			thrust::device_vector<float> other(count * count, 5);
  			thrust::device_vector<float> result(count * count, 0);

 			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

			sgemm(count, data, other, result);

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
  			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			printf("%d - THRUST: %f ms\n", count, time);
			max[k] += time;

			count <<= 1;
			k++;
		}
		k = 0;
	}

	for(int i = 0; i < 11; i++)
	{
		cout << max[i] / 10 << endl;
		max[i] = 0;
	}

	for(int i = 0; i < 10; i++)
	{
		count = 1;
		while(count <= N)
		{	
			float *a;
			float *b;
			float *c;

			float *a_dev;
			float *b_dev;
			float *c_dev;

			CUDA_CHECK_RETURN(cudaMalloc((void **)&a_dev, count * count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&b_dev, count * count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&c_dev, count * count * sizeof(float)));
	
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&a, count * count * sizeof(float)));	
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&b, count * count *sizeof(float)));	
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&c, count * count * sizeof(float)));	

			for (int i = 0; i < count; i++) 
       			{
				for(int j = 0; j < count; j++)
				{
          				a[i * count + j] = rand() % 1000;
	    				b[i * count + j] = rand() % 1000;
					c[i * count + j] = 0;
				}
			}

			CUDA_CHECK_RETURN(cublasSetMatrix(count, count, sizeof(float), (void *)a, count, (void *)a_dev, count));
			CUDA_CHECK_RETURN(cublasSetMatrix(count, count, sizeof(float), (void *)b, count, (void *)b_dev, count));
			CUDA_CHECK_RETURN(cublasSetMatrix(count, count, sizeof(float), (void *)c, count, (void *)c_dev, count));

			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
			CUDA_CHECK_RETURN(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, count, count, count, &alpha, a_dev, count, b_dev, count, &alpha, c_dev, count));

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

			CUDA_CHECK_RETURN(cublasGetMatrix(count, count, sizeof(float), (void *)a_dev, count, (void *)a, count));
			CUDA_CHECK_RETURN(cublasGetMatrix(count, count, sizeof(float), (void *)b_dev, count, (void *)b, count));
			CUDA_CHECK_RETURN(cublasGetMatrix(count, count, sizeof(float), (void *)c_dev, count, (void *)c, count));

			printf("%d - cuBLAS: %f ms\n", count, time);
			max[k] += time;

			CUDA_CHECK_RETURN(cudaFreeHost(a));
			CUDA_CHECK_RETURN(cudaFreeHost(b));
			CUDA_CHECK_RETURN(cudaFreeHost(c));

			CUDA_CHECK_RETURN(cudaFree(a_dev));
			CUDA_CHECK_RETURN(cudaFree(b_dev));
			CUDA_CHECK_RETURN(cudaFree(c_dev));

			count <<= 1;
			k++;
       	 	}
		k = 0;
	}

	for(int i = 0; i < 11; i++)
	{
		cout << max[i] / 10 << endl;
		max[i] = 0;
	}

	CUDA_CHECK_RETURN(cublasDestroy(cublas_handle));
	
	return 0;
}