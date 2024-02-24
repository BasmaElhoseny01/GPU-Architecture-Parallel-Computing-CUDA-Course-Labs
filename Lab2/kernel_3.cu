//  nvcc -o out  .\kernel_2.cu
// ./out.exe [in vs code]   out.exe in terminal
// nvprof out  [in terminal only]

// Vector addition in CUDA (Kernel2: level2 parallelism-> multiple blocks, multiple threads)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float* out,float* a,float* b,int n){
    // This Kernel is executed by every thread
    // Thread Id = thread_index + blick
    int tid=threadIdx.x + blockIdx.x* blockDim.x;

    // Handling arbitrary vector size
    if(tid<n){
        out[tid]=a[tid]+b[tid];
    }

    // Here since the no of blocks [Grid size is choosen based in the N] so every thread here is responsible for one element in the array :D
}


int main(){

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Calling Kernel
    int block_size=256; //each block will have 256 thread :D
    dim3 threadsPerBlock(16,16); // Threads per block are 256 arranged in array shape <3

    // What is the no of block??? Grid size?! 
    int grid_size= (N/block_size)+1;  //Ceil(N/block_size)

    // Total No of threads= grid_size*block_size  in above example 10000128 :D
    printf("Total no of threads %d\n",grid_size*block_size);
    vector_add<<<grid_size,block_size>>>(d_out,d_a,d_b,N);


    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a);
    free(b);
    free(out);

}