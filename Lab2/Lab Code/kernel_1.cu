//  nvcc -o out  .\kernel_1.cu
// ./out.exe [in vs code]   out.exe in terminal
// nvprof out  [in terminal only]

// Vector addition in pure C (CPU-only execution)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define N 10000000
#define MAX_ERR 1e-6
// Golabl Scope Func is on Device and can be called from [Host/Device]
__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i ++){
        out[i] = a[i] + b[i];
    }
}


int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // 2 Arraies to added and the result will be in array out
    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);


    // Initialize host arrays
    // a will be arary of 1s and b will be array of 2s
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    for(int i = 0; i < 5; i++){
        printf("%f %f\n",a[i],b[i]);
    }



    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // TransferData from host to device memory
    // Dst,src,size
    cudaMemcpy(d_a,a,sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,sizeof(float)*N,cudaMemcpyHostToDevice);

    // Executing the kernel 1block & 1thread no parallelism
    vector_add<<<1,1>>>(d_out,d_a,d_b,N);

    // Transfer data back from device to host memort
    cudaMemcpy(out,d_out,sizeof(float)*N,cudaMemcpyDeviceToHost);


    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");


    // Deallocate Device Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate Host Memory
    free(a);
    free(b);
    free(out);
 

    return 0;
}