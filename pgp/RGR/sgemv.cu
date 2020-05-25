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

const int N = 1 << 14;
#define V 0.2
#define T 2

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("")

void output(float* a, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%g ", a[i + j * n]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void sgemvblas(float *a, float *x, float *y, int count) 
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < count)
	{
		float res = 0;
		for(int j = 0; j < count; j++)
		{
			res += a[i * count + j] * x[j];
		}
		y[i] = res;
	}
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
      			sum += A[col * i + row] * B[i];			
		}
    		return sum;
	}
};

void sgemv(int count, thrust::device_vector<float> &data, thrust::device_vector<float> &other, thrust::device_vector<float> &result)
{
	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(count), result.begin(), dp(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(other.data()), count));
}

int main()
{
	srand(time(NULL));
	
	cudaEvent_t start, stop;

	float time = 0;
	float alpha = 1;
	float max[15];

	for(int i = 0; i < 15; i++)
	{
		max[i] = 0;
	}

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	cublasHandle_t cublas_handle;
	CUDA_CHECK_RETURN(cublasCreate(&cublas_handle)); 

	int count = 1;
	int k = 0;

	for(int i = 0; i < 10; i++)
	{
		count = 1;
		while(count <= N)
		{
			float *a;
			float *x;
			float *y;

			float *a_dev;
			float *x_dev;
			float *y_dev;

			CUDA_CHECK_RETURN(cudaMalloc((void **)&a_dev, count * count * sizeof(float)));	
			CUDA_CHECK_RETURN(cudaMalloc((void **)&x_dev, count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&y_dev, count * sizeof(float)));
	
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&a, count * count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&x, count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&y, count * sizeof(float)));

			for(int i = 0; i < count; i++)
			{
				x[i] = rand() % 1000;
				y[i] = 0;
				for(int j = 0; j < count; j++)
				{
					a[i * count + j] = rand() % 1000;	
				}
			}

    			CUDA_CHECK_RETURN(cudaMemcpy(a_dev, a, count * count * sizeof(float), cudaMemcpyHostToDevice));
    			CUDA_CHECK_RETURN(cudaMemcpy(x_dev, x, count * sizeof(float), cudaMemcpyHostToDevice));
    			CUDA_CHECK_RETURN(cudaMemcpy(y_dev, y, count * sizeof(float), cudaMemcpyHostToDevice));

			dim3 threadBlock(1024, 1);
			dim3 blockGrid(count / 1024 + 1, 1, 1);

       			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
			sgemvblas <<< blockGrid, threadBlock >>> (a_dev, x_dev, y_dev, count); 	
	
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

    			CUDA_CHECK_RETURN(cudaMemcpy(y, y_dev, count * sizeof(float), cudaMemcpyDeviceToHost));

			printf("%d - CUDA: %f ms\n", count, time);
			max[k] += time;

			CUDA_CHECK_RETURN(cudaFreeHost(a));
			CUDA_CHECK_RETURN(cudaFreeHost(x));
			CUDA_CHECK_RETURN(cudaFreeHost(y));

			CUDA_CHECK_RETURN(cudaFree(a_dev));
			CUDA_CHECK_RETURN(cudaFree(x_dev));
			CUDA_CHECK_RETURN(cudaFree(y_dev));

			count <<= 1;
			k++;
		}
		k = 0;
	}
	
	for(int i = 0; i < 15; i++)
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
  			thrust::device_vector<float> other(count, 5);
  			thrust::device_vector<float> result(count, 0);

 			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

  			sgemv(count, data, other, result);

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
	for(int i = 0; i < 15; i++)
	{
		cout << max[i] / 10 << endl;
		max[i] = 0;
	}

/////////////////////////////////////////////////////CUBLAS//////////////////////////////////////////////////////////////
	for(int i = 0; i < 10; i++)
	{
		count = 1;
		while(count <= N)
		{	
			float *a;
			float *x;
			float *y;

			float *a_dev;
			float *x_dev;
			float *y_dev;

			CUDA_CHECK_RETURN(cudaMalloc((void **)&a_dev, count * count * sizeof(float)));	
			CUDA_CHECK_RETURN(cudaMalloc((void **)&x_dev, count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc((void **)&y_dev, count * sizeof(float)));

			CUDA_CHECK_RETURN(cudaMallocHost((void **)&a, count * count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&x, count * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMallocHost((void **)&y, count * sizeof(float)));

			for(int i = 0; i < count; i++)
			{
				x[i] = rand() % 1000;
				y[i] = 0;
				for(int j = 0; j < count; j++)
				{
					a[i * count + j] = rand() % 1000;	
				}
			}
			CUDA_CHECK_RETURN(cublasSetMatrix(count, count, sizeof(float), (void *)a, count, (void *)a_dev, count));
			CUDA_CHECK_RETURN(cublasSetVector(count, sizeof(float), (void *)x, 1, (void *)x_dev, 1));
			CUDA_CHECK_RETURN(cublasSetVector(count, sizeof(float), (void *)y, 1, (void *)y_dev, 1));

			CUDA_CHECK_RETURN(cudaEventSynchronize(start));
			CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

			CUDA_CHECK_RETURN(cublasSgemv(cublas_handle, CUBLAS_OP_N, count, count, &alpha, a_dev, count, x_dev, 1, &alpha, y_dev, 1));

			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
			CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
			CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

			CUDA_CHECK_RETURN(cublasGetMatrix(count, count, sizeof(float), (void *)a_dev, count, (void *)a, count));
			CUDA_CHECK_RETURN(cublasGetVector(count, sizeof(float), (void *)x_dev, 1, (void *)x, 1));
			CUDA_CHECK_RETURN(cublasGetVector(count, sizeof(float), (void *)y_dev, 1, (void *)y, 1));

			printf("%d - cuBLAS: %f ms\n", count, time);
			max[k] += time;

			CUDA_CHECK_RETURN(cudaFreeHost(a));
			CUDA_CHECK_RETURN(cudaFreeHost(x));
			CUDA_CHECK_RETURN(cudaFreeHost(y));

			CUDA_CHECK_RETURN(cudaFree(a_dev));
			CUDA_CHECK_RETURN(cudaFree(x_dev));
			CUDA_CHECK_RETURN(cudaFree(y_dev));

			count <<= 1;
			k++;
        	}
		k = 0;
	}

	for(int i = 0; i < 15; i++)
	{
		cout << max[i] / 10 << endl;
		max[i] = 0;
	}

	CUDA_CHECK_RETURN(cublasDestroy(cublas_handle));
	
	return 0;
}