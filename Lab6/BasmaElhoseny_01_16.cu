#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

#define DEBUG 0

const int MAX_ARRAY_SIZE = 100000;
const int MAX_DEPTH = 3;

#define cucheck_dev(call)                                          \
    {                                                              \
        cudaError_t cucheck_err = (call);                          \
        if (cucheck_err != cudaSuccess)                            \
        {                                                          \
            const char *err_str = cudaGetErrorString(cucheck_err); \
            printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);  \
            assert(0);                                             \
        }                                                          \
    }

__device__ int g_uids = 0;

// Based on 1 block and 1024 threads :D
__global__ void binary_search_kernel(int N, float *d_array, int start_idx, int end_idx, float *d_target, int *d_target_index, int max_depth, int depth, int thread, int parent_uid)
{

    // We create a unique ID per block. Thread 0 does that and shares the value with the other threads.
    __shared__ int s_uid;
    if (threadIdx.x == 0)
    {
        s_uid = atomicAdd(&g_uids, 1);
        // printf("BLOCK %d launched by thread %d of block %d\n", s_uid, thread, parent_uid);
    }

    __syncthreads();

    // // Each Thread will check if number is in its part of the array or not
    int grid_tid = threadIdx.x + blockIdx.x * blockDim.x;

    // printf("grid_tid(%d):[%d,%d]\n", grid_tid, start_idx, end_idx);

    if (depth >= max_depth)
    {
        // We Can't Launch More Kernels :(
        // printf("Max Depth Reached We need to apply Binary Search\n");

        // Apply Binary Search by 2 Pointer :D
        int start = start_idx;
        int end = end_idx;

        while (start <= end)
        {
            // Calculate the middle index
            int mid = start + (end - start) / 2;

            if (d_array[mid] == *d_target)
            {
                // If the target is found at the middle index, update the target index and return
                *d_target_index = mid;

                // printf("Found at%d\n", *d_target_index);
                return;
            }
            else if (d_array[mid] < *d_target)
            {
                // If the target is greater than the value at the middle index,
                // update the start index to search in the right half
                start = mid + 1;
            }
            else
            {
                // If the target is smaller than the value at the middle index,
                // update the end index to search in the left half
                end = mid - 1;
            }
        }

        // we have reached loop wnd without return then not found by this range it dones't mean not found
        return;
    }

    // Split array into parts for each thread
    int N_range = end_idx - start_idx + 1;
    // printf("N_range:%d\n",N_range);
    int N_thread_range = N_range / blockDim.x; // floor
    // printf("N_thread_range:%d\n",N_thread_range);

    int th_start_index = start_idx + grid_tid * N_thread_range;
    int th_end_index = start_idx + (grid_tid + 1) * N_thread_range - 1; // inclusive

    // Last Thread Has to handle incomplete range
    if (grid_tid == blockDim.x - 1)
    {
        th_end_index = end_idx;
    }
    // printf("%d:[%d-%d]\n", grid_tid, th_start_index, th_end_index);

    // Launch New Kernel on the New Range
    binary_search_kernel<<<gridDim.x, blockDim.x>>>(N, d_array, th_start_index, th_end_index, d_target, d_target_index, max_depth, depth + 1, threadIdx.x, s_uid);
    cucheck_dev(cudaGetLastError());
    __syncthreads();

    return;
}

int seq_srearch(float *array, int array_size, float target_number)
{
    for (int i = 0; i < array_size; i++)
    {
        if (array[i] == target_number)
        {
            return i;
        }
    }
    return -1;
}

int main(int argc, char *argv[])
{
    // [TODO] Remove this
    // printf("Hello World\n");

    if (argc < 3)
    {
        printf("Usage: ./a.out <input_file> <number>\n");
        return 1;
    }

    // Read input file name from command line
    char *input_file = argv[1];

    // Read float number from command line
    float target_number = atof(argv[2]);

    // Read input file
    FILE *file = fopen(input_file, "r");

    // Check if file is opned correctly
    if (file == NULL)
    {
        printf("Error! opening file %s", input_file);
        // Program exits if the file pointer returns NULL.
        exit(1);
    }

    // Step(1) Allocate and init host Memory
    // Dynamic Array of Pointers
    float *array = (float *)malloc(MAX_ARRAY_SIZE * sizeof(float));

    int array_size = 0;
    while (fscanf(file, "%f", &array[array_size]) == 1)
    {
        array_size++;
    }

    // close file
    fclose(file);

    // Reallocation of memory
    array = (float *)realloc(array, array_size * sizeof(float));

    // // Print the array
    // for (int j = 0; j < array_size; j++)
    // {
    //     printf("%f\n", array[j]);
    // }

    // Step(2) Allocate device memory
    float *d_array;
    float *d_target;
    int *d_target_index;

    cudaMalloc((void **)&d_array, sizeof(float) * array_size);
    cudaMalloc((void **)&d_target, sizeof(float));
    cudaMalloc((void **)&d_target_index, sizeof(int));

    // Step(3) Transfer data from host to device memory
    cudaMemcpy(d_array, array, sizeof(float) * array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target_number, sizeof(float), cudaMemcpyHostToDevice);

    // Step(4) Calling Kernel
    int threads_per_block = 1024;

    // One Block Only
    int num_blocks = 1;

    // We set the recursion limit for CDP to max_depth.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    cudaMemset(d_target_index, -1, sizeof(int));

    binary_search_kernel<<<num_blocks, threads_per_block>>>(array_size, d_array, 0, array_size - 1, d_target, d_target_index, MAX_DEPTH, 1, 0, -1);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // in red
        printf("\033[1;31m");
        printf("CUDA error : %s\n", cudaGetErrorString(error));
        // reset color
        printf("\033[0m");
    }

    // Step(5) Transfer data from device to host memory int target_index;
    int *target_index = (int *)malloc(sizeof(int));
    cudaMemcpy(target_index, d_target_index, sizeof(int), cudaMemcpyDeviceToHost);

    // Step(6) Print the result
    printf("%d\n", *target_index);

    if (DEBUG)
    {
        // Sequential Search
        int seq_index = seq_srearch(array, array_size, target_number);
        if (seq_index != *target_index)
        {
            printf("Error: Sequential Search Result is not equal to Binary Search Result\n");
            printf("Sequential Search Result: %d\n", seq_index);
            printf("Binary Search Result: %d\n", *target_index);
        }
        else
        {
            printf("Success: Sequential Search Result is equal to Binary Search Result\n");
        }
    }

    return 0;
}

// nvcc -o out  ./k.cu
// ./out ./input.txt 6.4