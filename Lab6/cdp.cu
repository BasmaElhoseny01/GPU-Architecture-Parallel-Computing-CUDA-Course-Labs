#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
  }                                                         \
}

__device__ int g_uids = 0;

__global__ void cdp_kernel(int max_depth, int depth, int thread, int parent_uid, int* output)
{
    // We create a unique ID per block. Thread 0 does that and shares the value with the other threads.
    __shared__ int s_uid;
    if (threadIdx.x == 0)
    {
      s_uid = atomicAdd(&g_uids, 1);
      output[0] = s_uid;
      printf("BLOCK %d launched by thread %d of block %d\n", s_uid, thread, parent_uid);
    }

    __syncthreads();

    // We launch new blocks if we haven't reached the max_depth yet.
    if (depth >= max_depth)
      return;

    cdp_kernel<<<gridDim.x, blockDim.x>>>(max_depth, depth+1, threadIdx.x, s_uid, output);
    cucheck_dev(cudaGetLastError());
    __syncthreads();
}

int main(int argc, char **argv)
{
    printf("starting Simple Print (CUDA Dynamic Parallelism)\n");

    // Parse a few command-line arguments.
    int max_depth = 3;

    // Print a message describing what the sample does.

    printf("The CPU launches 2 blocks of 2 threads each. On the device each thread will\n");
    printf("launch 2 blocks of 2 threads each. The GPU we will do that recursively\n");
    printf("until it reaches max_depth=%d\n\n", max_depth);
    printf("In total 2");
    int num_blocks = 2, sum = 2;

    for (int i = 1 ; i < max_depth ; ++i)
    {
        num_blocks *= 4;
        printf("+%d", num_blocks);
        sum += num_blocks;
    }

    printf("=%d blocks are launched!!! (%d from the GPU)\n", sum, sum-2);
    printf("***************************************************************************\n\n");
    // We set the recursion limit for CDP to max_depth.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

    // Launch the kernel from the CPU.
    printf("Launching cdp_kernel() with CUDA Dynamic Parallelism:\n\n");
    int *output = (int *)malloc(sizeof(int)), *d_output;
    output[0] = 0;
    cudaMalloc((void **)&d_output, sizeof(int));
    cudaMemcpy(d_output, output, sizeof(int), cudaMemcpyHostToDevice);
    cdp_kernel<<<2, 2>>>(max_depth, 1, 0, -1, d_output);
    cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Last block id = %d\n", output[0]);
    exit(EXIT_SUCCESS);
}
