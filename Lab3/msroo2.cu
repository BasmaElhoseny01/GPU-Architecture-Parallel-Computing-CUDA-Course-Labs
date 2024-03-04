// kernel.cu : Defines the entry point for the console application.
//

// #include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
// #include <cutil.h>
#include <cuda_runtime.h>

#include <assert.h>

__device__ int get_index_to_check(int thread, int num_threads, int set_size, int offset)
{

    // Integer division trick to round up
    return (((set_size + num_threads) / num_threads) * thread) + offset;
}

__global__ void p_ary_search(int search, int array_length, int *arr, int *ret_val)
{

    const int num_threads = blockDim.x * gridDim.x;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;

    // ret_val[0] = -1;
    // ret_val[1] = offset;

    int set_size = array_length;

    while (set_size != 0)
    {
        // Get the offset of the array, initially set to 0
        int offset = ret_val[1];

        // I think this is necessary in case a thread gets ahead, and resets offset before it's read
        // This isn't necessary for the unit tests to pass, but I still like it here
        __syncthreads();

        // Get the next index to check
        int index_to_check = get_index_to_check(thread, num_threads, set_size, offset);

        // If the index is outside the bounds of the array then lets not check it
        if (index_to_check < array_length)
        {

            // If the next index is outside the bounds of the array, then set it to maximum array size
            int next_index_to_check = get_index_to_check(thread + 1, num_threads, set_size, offset);

            if (next_index_to_check >= array_length)
            {
                next_index_to_check = array_length - 1;
            }

            // If we're at the mid section of the array reset the offset to this index
            if (search > arr[index_to_check] && (search < arr[next_index_to_check]))
            {
                ret_val[1] = index_to_check;
            }
            else if (search == arr[index_to_check])
            {
                // Set the return var if we hit it
                ret_val[0] = index_to_check;
            }
        }

        // Since this is a p-ary search divide by our total threads to get the next set size
        set_size = set_size / num_threads;

        // Sync up so no threads jump ahead and get a bad offset
        __syncthreads();
    }
}

int chop_position(int search, int *search_array, int array_length)
{
    // Get the size of the array for future use
    int array_size = array_length * sizeof(int);

    // Don't bother with small arrays
    if (array_size == 0)
        return -1;

    // Setup array to use on device
    int *dev_arr;
    cudaMalloc((void **)&dev_arr, array_size);

    // Copy search array values
    cudaMemcpy(dev_arr, search_array, array_size, cudaMemcpyHostToDevice);

    // return values here and on device
    int *ret_val = (int *)malloc(sizeof(int) * 2);
    ret_val[0] = -1;                                                        // return value
    ret_val[1] = 0;                                                         // offset
    array_length = array_length % 2 == 0 ? array_length : array_length - 1; // array size

    int *dev_ret_val;
    cudaMalloc((void **)&dev_ret_val, sizeof(int) * 2);

    // Send in some intialized values
    cudaMemcpy(dev_ret_val, ret_val, sizeof(int) * 2, cudaMemcpyHostToDevice);

    // Launch kernel
    // This seems to be the best combo for p-ary search
    // Optimized around 10-15 registers per thread
    p_ary_search<<<16, 64>>>(search, array_length, dev_arr, dev_ret_val);

    // Get results
    cudaMemcpy(ret_val, dev_ret_val, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    int ret = ret_val[0];

    printf("Ret Val %i    Offset %i\n", ret, ret_val[1]);

    // Free memory on device
    cudaFree(dev_arr);
    cudaFree(dev_ret_val);

    free(ret_val);

    return ret;
}

// Test region
static int *build_array(int length)
{

    int *ret_val = (int *)malloc(length * sizeof(int));

    for (int i = 0; i < length; i++)
    {
        ret_val[i] = i * 2 - 1;
    }

    return ret_val;
}

static void test_array(int length, int search, int index)
{

    printf("Length %i   Search %i    Index %i\n", length, search, index);
    assert(index == chop_position(search, build_array(length), length) && "test_small_array()");
}

static void test_arrays()
{

    test_array(200, 200, -1);

    test_array(200, -1, 0);

    test_array(200, 1, 1);

    test_array(200, 29, 15);

    test_array(200, 129, 65);

    test_array(200, 395, 198);

    test_array(20000, 395, 198);

    test_array(2000000, 394, -1);

    test_array(20000000, 394, -1);
}

int main()
{
    test_arrays();
}

__global__ void binary_search(float *arr, float target, int *ans, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = 0;
    int right = n - 1;
    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target)
        {
            ans[tid] = mid;
            return;
        }
        else if (arr[mid] < target)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    ans[tid] = -1;
    __syncthreads();
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define THREADS 1024

__global__ void binary_search(int *arr, int target, int *ans, int N)
{
    __shared__ bool found;
    __shared__ int offset;
    offset = 0;
    found = false;
    __syncthreads();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int search_size = N;

    while (search_size && !found)
    {
        // Get the offset of the array, initially set to 0
        int t_amount = (search_size + THREADS - 1) / THREADS;
        int left = (t_amount * tid) + offset;
        // boundary check
        if (left < N)
        {
            // boundary check
            int right = min((t_amount * (tid + 1)) + offset, N - 1);

            if (target == arr[left])
            {
                ans[0] = left;
                found = true;
                return;
            }
            else if (target > arr[left] && (target < arr[right]))
            {
                offset = left;
            }
        }
        search_size /= THREADS;
        __syncthreads();
    }
}

__global__ void stupid_binary_search(int *arr, int target, int *ans, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = 0;
    int right = n - 1;
    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target)
        {
            ans[tid] = mid;
            return;
        }
        else if (arr[mid] < target)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    ans[tid] = -1;
    __syncthreads();
}

int main(int argc, char *argv[])
{

    // char *input_file = argv[1];
    // int target_value = atof(argv[2]);
    // Vector size
    int N = atoi(argv[1]);
    int search = atof(argv[2]);
    size_t bytes = N * sizeof(int);

    // Host data
    int *arr = (int *)malloc(bytes);
    // int *res = (int *)malloc(bytes);

    for (int i = 0; i < N; i++)
    {
        // random number
        arr[i] = rand() % N;
    }

    // sort the array
    qsort(arr, N, sizeof(int),
          [](const void *a, const void *b) -> int
          {
              return (*(int *)a - *(int *)b);
          });
    arr[N - 1] = 1027;

    // Allocate device memory
    int *d_arr;
    cudaMalloc((void **)&d_arr, bytes);

    // Copy to device
    cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice);

    int *ret_val = (int *)malloc(sizeof(int) * 2);
    ret_val[0] = -1; // return value
    ret_val[1] = 0;
    // int array_length = N;                                                   // offset
    // array_length = array_length % 2 == 0 ? array_length : array_length - 1; // array size

    int *dev_ret_val;
    cudaMalloc((void **)&dev_ret_val, sizeof(int) * 2);

    // Send in some intialized values
    cudaMemcpy(dev_ret_val, ret_val, sizeof(int) * 2, cudaMemcpyHostToDevice);

    // Launch the kernel
    binary_search<<<1, THREADS>>>(d_arr, search, dev_ret_val, N);

    // Get results
    cudaMemcpy(ret_val, dev_ret_val, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    int ret = ret_val[0];

    printf("Ret Val %i    Offset %i\n", ret, ret_val[1]);

    // print array
    // for (int i = 0; i < N; i++)
    // {
    //     if (arr[i] == search)
    //     {
    //         printf("index %i\n", i);
    //     }
    //     // printf("%i ", arr[i]);
    // }

    // check which ret_val2
    // for (int i = 0; i < N; i++)
    // {
    //     if (ret_val2[i] != -1)
    //     {
    //         printf("index %i \n", ret_val2[i]);
    //     }
    // }

    // write the array to a file
    FILE *f = fopen("array.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i = 0; i < N; i++)
    {
        fprintf(f, "%i\n", arr[i]);
    }
    return 0;
}