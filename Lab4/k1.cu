// nvcc -o out_1  ./k1.cu
// ./out_1 ./input ./output ./filter.txt
// nvprof out_2 ./testfile.txt ./out.txt


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>


#define MAX_ERR 1e-6

#define DIRENT_IMPLEMENTATION
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FILTER_DIM 3
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define IMAGE_CHANNELS 3

// Declare Constant Memory 
__constant__ float filter_c [FILTER_DIM][FILTER_DIM];

// Host Functions
__host__ float*  read_filter(const char* file_path){
    FILE* file = fopen(file_path, "r");
    if(file == NULL){
        printf("Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    int n;
    fscanf(file, "%d", &n);
    printf("Filter size: %d\n", n);

    printf("Filter Applied:\n");
    float* filter = (float*)malloc(n * n * sizeof(float));
    for(int i = 0; i < n * n; i++){
        fscanf(file, "%f", &filter[i]);
        printf("%.2f ", filter[i]);

        if (i % n == n-1){
            printf("\n");
        }
    }

    // Close File
    fclose(file);

    return filter;
}


__host__ unsigned char* read_image(const char* file_path){
    printf("Reading image at %s\n", file_path);

    int width, height, channels;
    unsigned char* image = stbi_load(file_path, &width, &height, &channels, 0); // Each group of three consecutive elements in the array represents the RGB values of one pixel.
    if(image == NULL){
        printf("Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    printf("Image size: %d x %d\n", width, height);
    printf("Channels: %d\n", channels);

    // assert (width == IMAGE_WIDTH);
    // assert (height == IMAGE_HEIGHT);
    // assert (channels == IMAGE_CHANNELS);

    return image;
}


void writeImageToFile(const char* filename, const unsigned char* imageData, int width, int height) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }

    // Write image width and height to file
    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    file.write(reinterpret_cast<const char*>(&height), sizeof(int));

    // Write image data to file
    file.write(reinterpret_cast<const char*>(imageData), width * height);

    file.close();
}

// Device Kernels
__global__ void BatchConvolution(unsigned char image[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS],unsigned char output_image[IMAGE_HEIGHT][IMAGE_WIDTH]){
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;

    //Boundary Cond
    if(outRow <IMAGE_HEIGHT  && outCol<IMAGE_WIDTH){
        float sum = 0;
        // Looping over mask :D
        for (int filterRow = 0; filterRow<FILTER_DIM; filterRow++){
            for (int filterCol = 0; filterCol<FILTER_DIM; filterCol++){
                int inRow = outRow - FILTER_DIM/2 + filterRow; // outRow - FilterRaduis + filterRow
                int inCol = outCol - FILTER_DIM/2 + filterCol; // outCol - FilterRaduis + filterCol

                // Check if out of Bounday --> This is useless in case of padding
                if (inRow>=0 && inRow <IMAGE_HEIGHT && inCol>=0 && inCol<IMAGE_WIDTH){
                    for(int c=0;c<3;c++){
                        // Every Channel
                        sum+=filter_c[filterRow][filterCol]*(float)image[inRow][inCol][c];
                    }    
                    // printf("%f.2    ",sum);               
                    // printf("%u\n",(unsigned int)sum);               
                }

            }
        }
        // printf("%d\n", (int)output_image[outRow][outCol]);
    output_image[outRow][outCol]=(unsigned char)sum;
    }

//    printf("Hello World\n");
//    printf("%d\n", (int)output_image[outRow][outCol]);
}


int main(int argc, char* argv[]){

    // Input Arguments 
    char* input_dir=argv[1];
    char* output_dir=argv[2];
    char* filter_pth=argv[3];

    // 1. Reading Filter
    float* filter=read_filter(filter_pth);

    // 2. initialize filter in constant memory
    cudaMemcpyToSymbol(filter_c,filter,FILTER_DIM*FILTER_DIM*sizeof(float));
    printf("Allocated Filter in Constant Memory\n");

    // 3. Ouptut Memory
    // 3.1. Allocate Host
    unsigned char output[IMAGE_HEIGHT][IMAGE_WIDTH];

    // 3.1. Allocate Device
    unsigned char* d_output;// Device pointer for the 2D array
    cudaMalloc((void**)&d_output, sizeof(unsigned char) * IMAGE_HEIGHT * IMAGE_WIDTH);


    // Open the input directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_dir)) != NULL) {
        printf("Reading Images from Directory: %s\n", input_dir);

        int index=0;

        // Iterate over each file in the directory
        while ((ent = readdir(dir)) != NULL) {
            // Filter out directories and special entries
            if (ent->d_type == DT_REG) {
                // Step(1) Read Image
                // Concatenate directory path and filename
                char file_path[256];
                snprintf(file_path, sizeof(file_path), "%s/%s", input_dir, ent->d_name);

                printf("Reading Image: %s\n", file_path);
                unsigned char* image_data = read_image(file_path);


                // statically allocate the matrices [To have as 3D Organzied]
                unsigned char image[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS];

                // Copy image to 3D array
                for (int y = 0; y < IMAGE_HEIGHT; ++y) {
                    for (int x = 0; x < IMAGE_WIDTH; ++x) {
                        for (int c = 0; c < IMAGE_CHANNELS; ++c) {
                            image[y][x][c] = image_data[(y * IMAGE_WIDTH + x) * IMAGE_CHANNELS + c];
                        }
                    }
                }
                // Empty the image_data memory [Flattened]
                stbi_image_free(image_data);


                // Check image Read
                // Write image data to JPEG file
                // stbi_write_jpg(ent->d_name, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, image, 90);

                // Step(2) Device Memory Allocation
                float *d_image;
                cudaMalloc((void**)&d_image, sizeof(unsigned char) * IMAGE_HEIGHT * IMAGE_WIDTH*IMAGE_CHANNELS);

                // Step(3) Copy from host to deivce
                cudaMemcpy(d_image, image, sizeof(unsigned char) * IMAGE_HEIGHT * IMAGE_WIDTH*IMAGE_CHANNELS, cudaMemcpyHostToDevice);

                // Step(4) Call Kernel
                // Kernel
                dim3 ThreadsPerBlock(16, 16);
                // dim3 ThreadsPerBlock(1, 1);
                dim3 GridSize ((IMAGE_WIDTH - 1) / ThreadsPerBlock.x + 1, (IMAGE_HEIGHT - 1) / ThreadsPerBlock.y + 1);

                // Call the kernel
                BatchConvolution<<<GridSize, ThreadsPerBlock>>>((unsigned char(*)[IMAGE_WIDTH][IMAGE_CHANNELS])d_image,(unsigned char(*)[IMAGE_WIDTH])d_output);

                // Step(5) Transfer data back to host memory
                cudaMemcpy(output, d_output, sizeof(unsigned char) * IMAGE_HEIGHT * IMAGE_WIDTH, cudaMemcpyDeviceToHost);

                // Save Image
                // Concatenate directory path and filename
                char out_file_path[256];
                snprintf(out_file_path, sizeof(out_file_path), "%s/%s", output_dir, ent->d_name);
                stbi_write_jpg(out_file_path, IMAGE_WIDTH, IMAGE_HEIGHT, 1, output, 90);


        //         // Verifcation
        //         // Perform convolution
        //         for (int i = 0; i < IMAGE_HEIGHT; i++) {
        //             for (int j = 0; j < IMAGE_HEIGHT; j++) {
        //                 unsigned char sum =0; // Initialize output at position (i,j) to zero
        //                 // Apply filter
        //                 for (int k = 0; k < FILTER_DIM; k++) {
        //                     for (int l = 0; l < FILTER_DIM; l++) {
        //                         int ni = i - FILTER_DIM / 2 + k;
        //                         int nj = j - FILTER_DIM / 2 + l;
        //                         for (int c = 0; c < IMAGE_CHANNELS; ++c) {
        //                             // Check boundaries
        //                             if (ni >= 0 && ni < IMAGE_HEIGHT && nj >= 0 && nj < IMAGE_WIDTH) {
        //                                 sum += image[ni][nj][c] * filter[k*FILTER_DIM + l];
        //                             }
        //                         }
        //                     }
        //                 }     
        //             printf("%d\n",sum);
        //             printf("%d\n",output[i][j]);
        //             assert(sum-output[i][j]<MAX_ERR);
        //             }
        //         }


                
        //         // Process the file here
        //         // Example: Load image using ent->d_name
            }
        }

        closedir(dir);
    }
    else {
        // Failed to open directory
        // Can;t open directory at path 
        perror("Failed to open Input directory");
        return EXIT_FAILURE;
    }

    // // Images as Batches
    // for(int batch_idx=0;batch_idx<batch_size;batch_idx++){
    //     // 2. Reading Image
    //     // unsigned char* image = read_image(file_path){

    //     // }


    //     // read_image(input_folder_pth);
    // }

}