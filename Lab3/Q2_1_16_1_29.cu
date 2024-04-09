#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define THREADS 1024
#define MAX_ERR 1e-6

__global__ void binary_search(float *arr, float target, int *ans, int N)
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

            // if (target == arr[left])
            if (abs(target - arr[left]) < MAX_ERR)
            {
                *ans = left;
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

int main(int argc, char *argv[])
{

    char *input_file = argv[1];
    float target_value = atof(argv[2]);

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

    fclose(file);

    // Allocate device memory
    float *d_arr;
    cudaMalloc((void **)&d_arr, bytes);

    // Copy to device
    cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice);

    int *ans = (int *)malloc(sizeof(int));
    *ans = -1; // return value

    int *dev_ans;
    cudaMalloc((void **)&dev_ans, sizeof(int));

    cudaMemcpy(dev_ans, ans, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    binary_search<<<1, THREADS>>>(d_arr, target_value, dev_ans, N);

    // Get results
    cudaMemcpy(ans, dev_ans, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%i", *ans);

    // Free memory
    free(arr);
    free(ans);
    cudaFree(d_arr);
    cudaFree(dev_ans);
    return 0;
}