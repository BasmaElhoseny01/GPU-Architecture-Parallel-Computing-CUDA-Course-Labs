#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define SIZE 256

__global__ void sum_vec(float *arr, float *res)
{
    // Allocate shared memory
    __shared__ int shared_sum[SIZE];

    // Calculate thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    shared_sum[threadIdx.x] = arr[tid];
    __syncthreads();

    // Iterate of log base 2 the block dimension
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = stride * threadIdx.x << 1;
        if (index < blockDim.x)
        {
            shared_sum[index] += shared_sum[index + stride];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
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
        arr[i] = 1;
    }
    // Allocate device memory
    float *d_arr, *d_res;
    cudaMalloc(&d_arr, bytes);
    cudaMalloc(&d_res, bytes);

    // Copy to device
    cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice);

    // TB Size
    const int TB_SIZE = 256;

    // Grid Size
    int GRID_SIZE = ceil(float(N) / TB_SIZE);

    // Print the grid size
    printf("Grid Size: %d\n", GRID_SIZE);
    // Call kernels
    sum_vec<<<GRID_SIZE, TB_SIZE>>>(d_arr, d_res);

    sum_vec<<<1, TB_SIZE>>>(d_res, d_res);

    // Copy to host;
    cudaMemcpy(res, d_res, bytes, cudaMemcpyDeviceToHost);

    // print the result

    printf("Sum: %f\n", res[0]);

    return 0;
}