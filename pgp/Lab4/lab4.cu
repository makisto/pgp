#include <cuda.h>
#include <stdio.h>

__global__ void gInitializeStorage(float* a)
{
    a[(threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x)] = 
    (float)((threadIdx.y + blockIdx.y * blockDim.y) + (threadIdx.x + blockIdx.x * blockDim.x) * (blockDim.x * gridDim.x));
}

__global__ void gTranspose0(float* a, float* b)
{
    b[(threadIdx.y + blockIdx.y * blockDim.y) + (threadIdx.x + blockIdx.x * blockDim.x) * (blockDim.x * gridDim.x)] 
    = a[(threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x)];
}

__global__ void gTranspose11(float* a, float* b)
{
    extern __shared__ float buffer[];
    buffer[threadIdx.y + threadIdx.x * blockDim.y] 
    = a[(threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x)];
    __syncthreads();
    b[(threadIdx.x + blockIdx.y * blockDim.x) + (threadIdx.y + blockIdx.x * blockDim.y) * (blockDim.x * gridDim.x)] 
    = buffer[threadIdx.x + threadIdx.y * blockDim.x];
}

#define SH_DIM 32 
__global__ void gTranspose12(float* a, float* b)
{
    __shared__ float buffer_s[SH_DIM][SH_DIM];
    buffer_s[threadIdx.y][threadIdx.x] 
    = a[(threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x)];
    __syncthreads();
    b[(threadIdx.x + blockIdx.y * blockDim.x) + (threadIdx.y + blockIdx.x * blockDim.y) * (blockDim.x * gridDim.x)]
     = buffer_s[threadIdx.x][threadIdx.y];
}

__global__ void gTranspose2(float* a, float* b)
{
    __shared__ float buffer[SH_DIM][SH_DIM + 1];
    buffer[threadIdx.y][threadIdx.x] 
    = a[(threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * (blockDim.x * gridDim.x)]; 
    __syncthreads();
    b[(threadIdx.x + blockIdx.y * blockDim.x) + (threadIdx.y + blockIdx.x * blockDim.y) * (blockDim.x * gridDim.x)] 
    = buffer[threadIdx.x][threadIdx.y];
}

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

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("") 

int main()
{
    int n, k;
    printf("Matrix size:\n");
    scanf("%d", &n);
    printf("Threads per block:\n");
    scanf("%d", &k);
    
    if(n % k)
    {
        fprintf(stderr, "change dimensions\n");
        return -1;
    }
    int b = n / k;
    const int max_size = 1 << 8;
    if(b > max_size)
    {
        fprintf(stderr, "too many blocks\n");
        return -1;
    }
    float* a, *a1, *c;
    
    CUDA_CHECK_RETURN(cudaMalloc((void**)&a, n * n * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&a1, n * n * sizeof(float)));    
    c = new float[n * n];

    gInitializeStorage <<< dim3(b, b), dim3(k, k) >>> (a);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(c, a, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);

    gTranspose0 <<< dim3(b, b), dim3(k, k) >>> (a, a1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(c, a1, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);
  
    gTranspose11 <<< dim3(b, b), dim3(k, k), k * k * sizeof(float) >>> (a, a1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    CUDA_CHECK_RETURN(cudaMemcpy(c, a1, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);
  
    gTranspose12 <<< dim3(b, b), dim3(k, k) >>> (a, a1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    CUDA_CHECK_RETURN(cudaMemcpy(c, a1, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);
   
    gTranspose2 <<< dim3(b, b), dim3(k, k) >>> (a, a1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    CUDA_CHECK_RETURN(cudaMemcpy(c, a1, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    //output(c, n);
  
    CUDA_CHECK_RETURN(cudaFree(a));
    CUDA_CHECK_RETURN(cudaFree(a1));
    free(c);

    return 0;
} 
