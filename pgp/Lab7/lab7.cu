#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <iostream>

using namespace std;

#define N 10
#define V 0.2
#define T 2

#define CUDA_CHECK_RETURN(value) ((cudaError_t)value != cudaSuccess) ? printf("Error %s at line %d in the file %s\n", cudaGetErrorString((cudaError_t)value), __LINE__, __FILE__) : printf("")

struct functor 
{
    const float koef;
    functor(float _koef) : koef(_koef){}
    __host__ __device__ float operator()(float x, float y) 
    { 
        return y + koef * (x - y);
    }
};

void iteration(float _koef, thrust::device_vector<float> &x, thrust::device_vector<float> &y)
{
    functor func(_koef);
    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}

__global__ void kernel(float *f, float *res) 
{
    int cur = threadIdx.x + blockDim.x * blockIdx.x;
    int prev = cur - 1;
    if(prev == -1)
    {
        prev = N - 1;	
    }
    res[cur] = f[cur] + (V * T) * (f[prev] - f[cur]);
}

int main()
{
    float *Function = new float[N];
    float *FunctionData = new float[N];
    float *frez;
    float *tempa;

    cudaEvent_t start, stop;
    float time;

    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));

    for (int i = 0; i < N; i++) 
    {
        FunctionData[i] = rand() % 100;
	Function[i] = FunctionData[i];
    }
    
    CUDA_CHECK_RETURN(cudaMalloc((void **)&frez, N * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&tempa, N * sizeof(float)));

    CUDA_CHECK_RETURN(cudaMemcpy(tempa, Function, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaEventSynchronize(start));
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

    for(int i = 0; i < 1000; i++)
    {
    	kernel <<< 1, N >>> (tempa, frez);
    	CUDA_CHECK_RETURN(cudaMemcpy(Function, frez, N * sizeof(float), cudaMemcpyDeviceToHost));
        //CUDA_CHECK_RETURN(cudaMemcpy(tempa, frez, N * sizeof(float), cudaMemcpyHostToDevice));
        /*for(int i = 0; i < N; i++)
        {
            cout << Function[i] << " ";
        }   
        cout << endl;*/
    }

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("CUDA: %f ms\n", time);
    
    thrust::host_vector<float> cpumem1(N);
    thrust::host_vector<float> cpumem2(N);

    for (int i = 0; i < N; i++)
    {
        cpumem1[i] = FunctionData[i];
        if(i - 1 >= 0)
        {
            cpumem2[i] = FunctionData[i - 1];
        }
        else
        {
            cpumem2[i] = FunctionData[N - 1];
        }
    }

    thrust::device_vector<float> gpumem1 = cpumem1;
    thrust::device_vector<float> gpumem2 = cpumem2;
       
    CUDA_CHECK_RETURN(cudaEventSynchronize(start));
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
  
    for(int i = 0; i < 1000; i++)
    { 
        iteration(V * T, gpumem2, gpumem1);
        /*for(int i = 0; i < N; i++)
        {
            cout << gpumem1[i] << " ";
        }
        cout << endl;*/
        thrust::copy(gpumem1.begin(), gpumem1.end(), gpumem2.begin());
       /* thrust::copy(cpumem1.begin(), cpumem1.end(), gpumem1.begin());
        thrust::copy(cpumem2.begin(), cpumem2.end(), gpumem2.begin());*/
        /*for(int i = 0; i < N; i++)
        {
            if(i - 1 >= 0)
            {
                gpumem2[i] = gpumem1[i - 1];
            }
            else
            {
                gpumem2[i] = gpumem1[N - 1];
            }
        }*/
    }
    
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&time, start, stop));
    printf("Thrust: %f ms\n", time);

    return 0;
}

