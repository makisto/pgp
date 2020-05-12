#include <cuda.h>
#include <stdio.h>

__global__ void sum(float* a, float* b, float* c)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x; 
    //int index = blockIdx.x;
    c[index] = a[index] + b[index];
}

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("") 

int main()
{ 
    int n, k;
    scanf("%d%d", &n, &k);

    float* a = new float[n * k];
    float* b = new float[n * k];
    float* c = new float[n * k];

    for(int i = 0; i < n * k; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    float* dev1;
    float* dev2;
    float* dev3;

    float elapsedTime;
    cudaEvent_t start, stop;

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev1, n * k * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev2, n * k * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev3, n * k * sizeof(float)));

    CUDA_CHECK_RETURN(cudaMemcpy(dev1, a, n * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(dev2, b, n * k * sizeof(float), cudaMemcpyHostToDevice));
   
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
    sum <<< n, k >>> (dev1, dev2, dev3);
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

    CUDA_CHECK_RETURN(cudaGetLastError());

    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
  
    fprintf(stderr, "gTest took %g\n", elapsedTime);

    CUDA_CHECK_RETURN(cudaEventDestroy(start));
    CUDA_CHECK_RETURN(cudaEventDestroy(stop));

    CUDA_CHECK_RETURN(cudaMemcpy(c, dev3, n * k * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = (n * k) - 5; i < n * k; i++)
    {
        printf("Element #%i: %f\n", i, c[i]);
    }

    free(a);
    free(b);
    free(c);

    CUDA_CHECK_RETURN(cudaFree(dev1));
    CUDA_CHECK_RETURN(cudaFree(dev2));
    CUDA_CHECK_RETURN(cudaFree(dev3));

    return 0;
}
