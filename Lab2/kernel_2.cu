//  nvcc -o out  .\kernel_2.cu
// ./out.exe [in vs code]   out.exe in terminal
// nvprof out  [in terminal only]

// Vector addition in CUDA (Kernel2: level1 parallelism-> 1block, multiple threads)


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#define N 10000000
#define MAX_ERR 1e-6


__global__ void vector_add(float * out,float *a,float *b,int n){
    // This function is executed by each thread :D
    // Thread index
    int index=threadIdx.x;

    // Block dim
    int stride=blockDim.x;

    for (int i = index; i<n; i+=stride){
        out[i]=a[i]+b[i];
    }
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

    // Allocate Device Memory
    cudaMalloc((void**)&d_a,sizeof(float)*N);
    cudaMalloc((void**)&d_b,sizeof(float)*N);
    cudaMalloc((void**)&d_out,sizeof(float)*N);

    // Transfer Data from host to device memory
    // dts,src,size
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


    // Executing kernel (1block, 256 thread: no parallelism)
    vector_add<<<1,256>>>(d_out, d_a, d_b, N);

    // Transfer Data back to host Memory
    // Sync API code is blocked till d_out is avaliable
    cudaMemcpy(out,d_out,sizeof(float)*N,cudaMemcpyDeviceToHost);

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




    
    return 0;
}