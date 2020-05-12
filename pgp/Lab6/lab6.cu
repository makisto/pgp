#include <cuda.h>
#include <ctime>
#include <stdio.h>
#include <iostream>

int K = 256;
int N = 1024 * 32;
int sizeVector = (N * 32 * 20);

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("")

__global__ void addKernel(int *a, int *b, int *c)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	c[i] = a[i] + b[i];
}

__global__ void mulKernel(int *a, int *b, int *c)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	c[i] = a[i] * b[i];
}

void VectorOps()
{
	cudaStream_t stream0, stream1;
	int *a;
	int *b;
	int *c;

	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&a, sizeVector * sizeof(int), cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&b, sizeVector * sizeof(int), cudaHostAllocDefault));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&c, sizeVector * sizeof(int), cudaHostAllocDefault));

        for(int i = 0; i < sizeVector; i++)
	{
		a[i] = rand() % 20;
		b[i] = rand() % 20;
	}
        
       	int *dev_a0 = 0;
	int *dev_b0 = 0;
	int *dev_c0 = 0;

	int *dev_a1 = 0;
	int *dev_b1 = 0;
	int *dev_c1 = 0;

	cudaEvent_t start, stop;

	CUDA_CHECK_RETURN(cudaStreamCreate(&stream0));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	
	float time;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_c0, sizeVector * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a0, sizeVector * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_b0, sizeVector * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_c1, sizeVector * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a1, sizeVector * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_b1, sizeVector * sizeof(int)));

        CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
	for(int i = 0; i < sizeVector; i += N)
	{
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a0, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b0, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));

	    addKernel <<< N / K, K, 0, stream0 >>> (dev_a0, dev_b0, dev_c0);
	    
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
	}

	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream0));
	
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

	printf("\nV.1-Add Time: %f ms\n", time);

        for(int i = sizeVector - 10; i < sizeVector; i++)
	{
		std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
	}
	
	for(int i = 0; i < sizeVector; i++)
	{
		a[i] = rand() % 20;
		b[i] = rand() % 20;
	}

	CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
        for(int i = 0; i < sizeVector; i += N)
	{
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a1, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b1, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

	    mulKernel <<< N / K, K, 0, stream1 >>> (dev_a1, dev_b1, dev_c1);

	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

	printf("\nV.1-Mul Time: %f ms\n", time);

        for(int i = sizeVector - 10; i < sizeVector; i++)
	{
		std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
	}
	
	for(int i = 0; i < sizeVector; i++)
	{
		a[i] = rand() % 20;
		b[i] = rand() % 20;
	}

	CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
	for(int i = 0; i < sizeVector; i += N * 2)
	{
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a0, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b0, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));

	    addKernel <<< N / K, K, 0, stream0 >>> (dev_a0, dev_b0, dev_c0);
	    
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));

            CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a1, a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b1, b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

   	    addKernel <<< N / K, K, 0, stream1 >>> (dev_a1, dev_b1, dev_c1);

	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream0));
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

	printf("\nV.2-Add Time: %f ms\n", time);

        for(int i = sizeVector - 10; i < sizeVector; i++)
	{
		std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
	}
	
	for(int i = 0; i < sizeVector; i++)
	{
		a[i] = rand() % 20;
		b[i] = rand() % 20;
	}

	CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
        for(int i = 0; i < sizeVector; i += N * 2)
	{
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a0, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b0, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));

	    mulKernel <<< N / K, K, 0, stream0 >>> (dev_a0, dev_b0, dev_c0);
	    
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));

            CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a1, a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b1, b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

   	    mulKernel <<< N / K, K, 0, stream1 >>> (dev_a1, dev_b1, dev_c1);

	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream0));
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

	printf("\nV.2-Mul Time: %f ms\n", time);

        for(int i = sizeVector - 10; i < sizeVector; i++)
	{
		std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
	}
	
	for(int i = 0; i < sizeVector; i++)
	{
		a[i] = rand() % 20;
		b[i] = rand() % 20;
	}

	CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
        for(int i = 0; i < sizeVector; i += N * 2)
	{
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a0, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a1, a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b0, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b1, b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

	    addKernel <<< N / K, K, 0, stream0 >>> (dev_a0, dev_b0, dev_c0);
	    addKernel <<< N / K, K, 0, stream1 >>> (dev_a1, dev_b1, dev_c1);
	    
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream0));
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

	printf("\nV.3-Add Time: %f ms\n", time);

        for(int i = sizeVector - 10; i < sizeVector; i++)
	{
		std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
	}

	for(int i = 0; i < sizeVector; i++)
	{
		a[i] = rand() % 20;
		b[i] = rand() % 20;
	}

	CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	
	for(int i = 0; i < sizeVector; i += N * 2)
	{
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a0, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a1, a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b0, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b1, b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));
	    
	    mulKernel <<< N / K, K, 0, stream0 >>> (dev_a0, dev_b0, dev_c0);
            mulKernel <<< N / K, K, 0, stream1 >>> (dev_a1, dev_b1, dev_c1);
	    
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
	    CUDA_CHECK_RETURN(cudaMemcpyAsync(c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream0));
	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));

	printf("\nV.3-Mul Time: %f ms\n", time);

        for(int i = sizeVector - 10; i < sizeVector; i++)
	{
		std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
	}

        CUDA_CHECK_RETURN(cudaFreeHost(a));
	CUDA_CHECK_RETURN(cudaFreeHost(b));
	CUDA_CHECK_RETURN(cudaFreeHost(c));

	CUDA_CHECK_RETURN(cudaFree(dev_a0));
	CUDA_CHECK_RETURN(cudaFree(dev_a1));
	CUDA_CHECK_RETURN(cudaFree(dev_b0));
	CUDA_CHECK_RETURN(cudaFree(dev_b1));
	CUDA_CHECK_RETURN(cudaFree(dev_c0));
	CUDA_CHECK_RETURN(cudaFree(dev_c1));
}

float cuda_memory_malloc_test(int size, bool up)
{
	cudaEvent_t start, stop;

	int *a, *dev_a;
	float elapsedTime = 0.0f;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	a = (int*)malloc(size * sizeof(*a));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));
	
	CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	for (int i = 0; i < 100; i++) 
	{
		if(up)
		{
	    		CUDA_CHECK_RETURN(cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
		}
		else
		{
	    		CUDA_CHECK_RETURN(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
		}
	}

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));

	free(a);

	CUDA_CHECK_RETURN(cudaFree(dev_a));
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));

	return elapsedTime;
}

float cuda_alloc_memory_malloc_test(int size, bool up)
{
	cudaEvent_t start, stop;
	int *a, *dev_a;
	float elapsedTime = 0.0f;

	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));
	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&a, size * sizeof(*a),cudaHostAllocDefault));

	CUDA_CHECK_RETURN(cudaEventSynchronize(start));
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

	for (int i = 0; i < 100; i++) 
	{
		if (up)
		{
	    		CUDA_CHECK_RETURN(cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
		}
		else
        	{
	    		CUDA_CHECK_RETURN(cudaMemcpy(a, dev_a, size * sizeof(*a), cudaMemcpyDeviceToHost));
		}
	}	

	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));

	CUDA_CHECK_RETURN(cudaFree(dev_a));
	CUDA_CHECK_RETURN(cudaFreeHost(a));
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));

	return elapsedTime;
} 

int main()
{
	srand(time(NULL));

	float elapsedTime;
	float MB = (float)100 * sizeVector * sizeof(int)/1024/1024;
	
	elapsedTime = cuda_memory_malloc_test(sizeVector, true);
	printf("Without block pages GPU: %3.5f ms\n",elapsedTime);
	printf("\tMB GPU %3.1f\n", MB/(elapsedTime/1000));

	elapsedTime = cuda_memory_malloc_test(sizeVector, false);
	printf("Without block pages CPU: %3.5f ms\n", elapsedTime);
	printf("\tMB CPU %3.1f\n", MB / (elapsedTime / 1000));
	
	elapsedTime = cuda_alloc_memory_malloc_test(sizeVector, true);
	printf("Block pages GPU: %3.5f ms\n", elapsedTime);
	printf("\tMB GPU %3.1f\n", MB / (elapsedTime / 1000));

	elapsedTime = cuda_alloc_memory_malloc_test(sizeVector, false);
	printf("Block pages CPU: %3.5f ms\n", elapsedTime);
	printf("\tMB CPU %3.1f\n", MB / (elapsedTime / 1000));
	
	VectorOps();

	return 0;
}
