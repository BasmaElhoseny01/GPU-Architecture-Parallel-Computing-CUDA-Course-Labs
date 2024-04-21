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

    char *input_file = argv[1];
    // float target_value = atof(argv[2]);

    int N = 0;

    // Read the file
    FILE *file = fopen(input_file, "r");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    // Count the number of lines
    char ch;
    while (!feof(file))
    {
        ch = fgetc(file);
        if (ch == '\n')
        {
            N++;
        }
    }
    fclose(file);

    // Allocate memory
    size_t bytes = N * sizeof(float);
    float *arr = (float *)malloc(bytes);

    // Read the file
    file = fopen(input_file, "r");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i = 0; i < N; i++)
    {
        fscanf(file, "%f", &arr[i]);
    }

    // Host data
    float *res = (float *)malloc(bytes);

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