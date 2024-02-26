// nvcc -o out_2  ./k2.cu
// out_2 ./testfile.txt ./out.txt
// nvprof out_2 ./testfile.txt ./out.txt

// Matrix Additon Level1 parallelism (1block, multiple threads)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define MAX_ERR 1e-6




void skip_comments(FILE * file){
    char c;
    // Consume non-numeric characters until a newline or end of file
    while ((c = fgetc(file)) != EOF && c != '\n') {
        if (c == '/' && (c = fgetc(file)) == '/') { // Handle comments
            while ((c = fgetc(file)) != '\n' && c != EOF); // Skip the comment line
            break; // Exit inner loop
        }
    }
    return;

}

void read_matrices(FILE* file, const int rows,const int cols ,float* a,float* b) {
    //[row,col] --> flattening col+row*width
    // 2. Read values row by row [Matrix 1]
    for (int i=0;i<rows;i++){
        //3. Read col by col
        for (int j=0;j<cols;j++){
            float num;
            if (fscanf(file,"%f", &num) != 1) {
                printf("Invalid input. Please enter a float num. \n");
                exit(1);    
            }
            // printf("%f\n",num);
            a[j+i*cols]=num;
        }
        skip_comments(file);
    }

    // 2. Read values row by row [Matrix 2]
    for (int i=0;i<rows;i++){
        //3. Read col by col
        for (int j=0;j<cols;j++){
            float num;
            if (fscanf(file,"%f", &num) != 1) {
                printf("Invalid input. Please enter a float num. \n");
                exit(1);    
            }
            // printf("%f\n",num);
            b[j+i*cols]=num;
        }
        skip_comments(file);
    }

}


void write_matrix(FILE*file,float*matrix,const int rows,const int cols){
    for (int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            float num=matrix[j+i*cols];
            fprintf(file, "%f",num);

            // Add space if not end
            if(j<cols-1){
              fprintf(file," ");
            }
        }
        fprintf(file,"\n");

    }
}


__global__ void add_matrix(float*a,float*b,float*c,const int rows,const int cols){
    // This Kernel is executed by every thread
    int tid=threadIdx.x+blockIdx.x*blockDim.x;

    // Computing Row Values
    // Conditon on only avaliable rows :D
    if(tid<rows){
        for(int j=0;j<cols;j++){
            c[j+tid*cols]=a[j+tid*cols]+b[j+tid*cols];
            printf("c=%f: a+b=%f+%f\n",c[j+tid*cols],a[j+tid*cols], b[j+tid*cols]);
        }
    }
   
    return;
}


int main(int argc, char* argv[]){
    printf("Kernel(2)\n");


    // Redaing txt files
    char* input_pth=argv[1];
    char* out_pth=argv[2];

    printf("Input:%s\n",input_pth);
    printf("Output:%s\n",out_pth);


    // Opening input file
    FILE* file = fopen (input_pth, "r");
    FILE* file_out = fopen (out_pth, "w");

    // Check if file is opned correctly
    if (file == NULL){
        printf("Error! opening file %s",input_pth);
        // Program exits if the file pointer returns NULL.
        exit(1);
    }

    // Check if file is opned correctly
    if (file_out == NULL){
        printf("Error! opening file %s",out_pth);
        // Program exits if the file pointer returns NULL.
        exit(1);
    }

    // Reading file content
    int test_cases_no;
    if ( fscanf(file, "%d", &test_cases_no)==EOF){
        printf("Error in reading int no of testcases in the first line");
        exit(1);    
    }
    printf("Total No of Test Cases %d\n",test_cases_no);
    skip_comments(file);

    while(test_cases_no>0){
        // For each test case
        // 1. Read no of rows and cols
        int rows,cols;

        if (fscanf(file,"%d %d", &rows, &cols) != 2) {
            printf("Invalid input. Please enter two integers. to represnt rows and cols respectively \n");
            exit(1);    
        }
        printf("rows:%d-cols:%d\n",rows,cols);
        skip_comments(file);


        // Step(1) Allocate and init host Memory
        // Dynamic Array of Pointers
        float *a = (float *)malloc(rows*cols * sizeof(float));
        float *b = (float *)malloc(rows*cols * sizeof(float));
        float *c = (float *)malloc(rows*cols * sizeof(float));

        // Read Matric values from file
        read_matrices(file,rows,cols,a,b);


        // Step(2) Allocate device memory
        float *d_a, *d_b, *d_c;
        cudaMalloc((void**)&d_a, sizeof(float) * rows*cols);
        cudaMalloc((void**)&d_b, sizeof(float) * rows*cols);
        cudaMalloc((void**)&d_c, sizeof(float) * rows*cols);


        // Step(3) Transfer data from host to device memory
        cudaMemcpy(d_a,a, sizeof(float)*rows*cols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b,b, sizeof(float)*rows*cols, cudaMemcpyHostToDevice);


        // Step(4) Calling Kernel
        // Block size with 16*16 is the common choice
        int block_size=256; // no of threads per block
        // Since we need one thread per row ceil(rows-1/256)
        int grid_size=((rows-1)/block_size)+1;

        //For the test case we have only 3 rows so only 1 block is very redundent 
        //  We will have 256 thread
        printf("Total no of blocks %d\n",grid_size);
        printf("Total no of threads %d\n",block_size*grid_size);

        add_matrix<<<grid_size,block_size>>>(d_a,d_b,d_c,rows,cols);
        
        // Step(5) Transfer data back to host memory
        cudaMemcpy(c,d_c,sizeof(float)*rows*cols,cudaMemcpyDeviceToHost);

        

        // // Verification :D
        // for(int i = 0; i < rows; i++){
        //     for(int j = 0; j < cols; j++){
        //         assert(fabs(c[j+i*cols] - a[j+i*cols] - b[j+i*cols]) < MAX_ERR);
        //     }
        // }

        // printf("PASSED\n");

        // Write Result to the file
        write_matrix(file_out,c,rows,cols);


        // Step(6) Free Device Memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        // Step(7) Free Host Memory
        free(a);
        free(b);
        free(c);

        test_cases_no--;
        // test_cases_no=0;
        printf("\n\n");
    }



    fclose(file); // Close the file
    fclose(file_out); // Close the file
    return 0;
}