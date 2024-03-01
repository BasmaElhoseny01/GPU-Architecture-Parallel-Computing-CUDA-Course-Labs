// nvcc -o out_1  ./k1.cu
// out_1 ./input.txt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAX_ERR 1e-6

#define SIZE 256

void printFloatArray(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f\n", arr[i]);
    }
}

void read_vector(FILE*file,float*arr,const int size){    
    // Read floats until the end of file
    float num;
    for (int i=0;i<size;i++){
        if (fscanf(file,"%f", &num) != 1) {
            printf("Invalid input. Please enter a float num. \n");
            exit(1);    
        }
        arr[i]=num;
    }
}

__global__ void add_vector(float*arr,const int size){
    printf("threadIdx.x %d\n",threadIdx.x);
    //Shared memory is a fast on-chip memory accessible to all threads within the same block.
    //Each block has its own shared memory space, and data stored in shared memory can be efficiently shared and accessed by threads within the block.
    __shared__ float partialSum[SIZE];

    // The partial sum related to that thread then we will add the first eleent to that partial sum
    partialSum[threadIdx.x] = arr[threadIdx.x+blockDim.x*blockIdx.x];

    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        // Ensure all threads a have completed 
        __syncthreads();
        if(threadIdx.x % (2*stride) == 0)
            partialSum[threadIdx.x ] += partialSum[threadIdx.x+stride];    
    }
    return;
}

int main(int argc, char* argv[]){
    printf("Kernel(1)\n");


    // Redaing txt file
    char* input_pth=argv[1];
    int size;


    // Opening input file
    FILE * file = fopen(input_pth, "r");

    // Check if the file opened successfully
    if (file == NULL) {
        printf("Unable to open the file.\n");
        return 1;
    }

    // Reading file content
    if (fscanf(file, "%d", &size)==EOF){
        printf("Error in reading int size of array in the first line");
        exit(1);    
    }
    printf("%d\n",size);

    // Step(1) Allocate and init host Memory
    float *arr = (float *)malloc(size * sizeof(float));

    // Read array vakes from the file
    read_vector(file,arr,size);
    // printFloatArray(arr,size);

    // Step(2) Allocate device memory
    float *d_arr;
    cudaMalloc((void**)&d_arr, sizeof(float) * size);

    // Step(3) Transfer data from host to device memory
    cudaMemcpy(d_arr,arr, sizeof(float)*size, cudaMemcpyHostToDevice);



    // Step(4) Calling Kernel
    // Block size with 16*1 is the common choice
    add_vector<<<1,size>>>(d_arr,size);

     
    // Step(5) Transfer data back to host memory
    // cudaMemcpy(c,d_ar,sizeof(float)*size,cudaMemcpyDeviceToHost);

    // Print Results


    // Step(6) Free Device Memory
    cudaFree(d_arr);

    // Step(7) Free Host Memory
    free(arr);   

    return 0;
}