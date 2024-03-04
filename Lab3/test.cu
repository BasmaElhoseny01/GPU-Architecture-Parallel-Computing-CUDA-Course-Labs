#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define SIZE 256

__global__ void sum_vec(float *arr, float *res, int N)
{
    // Allocate shared memory
    __shared__ float shared_sum[SIZE];

    // Calculate thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int amount_per_thread = ceil(float(N) / blockDim.x);

    int start_index = tid * amount_per_thread;

    int end_index = min(start_index + amount_per_thread, N);

    for (int k = start_index; k < end_index; k++)
    {
        shared_sum[tid] += arr[k];
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = stride * threadIdx.x << 1;
        if (index < blockDim.x)
        {
            shared_sum[index] += shared_sum[index + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        res[blockIdx.x] = shared_sum[0];
    }
}

int main(int argc, char *argv[])
{

    // Vector size
    int N = atoi(argv[1]);
    size_t bytes = N * sizeof(float);

    // Host data
    float *arr = (float *)malloc(bytes);
    float *res = (float *)malloc(bytes);

    for (int i = 0; i < N; i++)
    {
        // arr[i] = atof(argv[i + 2]);
        arr[i] = 1.0f;
    }
    // Allocate device memory
    float *d_arr, *d_res;
    cudaMalloc(&d_arr, bytes);
    cudaMalloc(&d_res, bytes);

    // Copy to device
    cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice);

    // Call kernels
    sum_vec<<<1, SIZE>>>(d_arr, d_res, N);

    // Copy to host;
    cudaMemcpy(res, d_res, bytes, cudaMemcpyDeviceToHost);

    // print the result
    printf("%f\n", res[0]);

    cudaFree(d_arr);
    cudaFree(d_res);

    free(arr);
    free(res);

    return 0;
}