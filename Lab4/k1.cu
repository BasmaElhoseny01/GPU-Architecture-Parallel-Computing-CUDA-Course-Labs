// nvcc -o out_1  ./k1.cu
// ./out_1 ./input ./output ./filters/avg_9_9.txt
// nvprof out_2 ./testfile.txt ./out.txt

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string.h>

#define MAX_ERR 1e-6

#define DIRENT_IMPLEMENTATION
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_CHANNELS 3

// Declare Constant Memory
// Max is 400 floating element :D 
__constant__ float filter_c[20 * 20];

// Host Functions
__host__ float *read_filter(const char *file_path,int &filter_dim)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        printf("Error: Unable to open file %s\n", file_path);
        exit(1);
    }


    fscanf(file, "%d", &filter_dim);
    printf("Filter size: %d\n", filter_dim);

    printf("Filter Applied:\n");
    float *filter = (float *)malloc(filter_dim * filter_dim * sizeof(float));
    for (int i = 0; i < filter_dim * filter_dim; i++)
    {
        fscanf(file, "%f", &filter[i]);
        printf("%f ", filter[i]);

        if (i % filter_dim == filter_dim - 1)
        {
            printf("\n");
        }
    }

    // Close File
    fclose(file);

    return filter;
}

__host__ void read_image(const char *filename, float **data, int *width, int *height, int *channels)
{
    //Read Image
    unsigned char *udata = stbi_load(filename, width, height, channels, 0);

    // Host Memory Allocation & convert data from unsigned char to float
    *data = (float *)malloc((*width) * (*height) * (*channels) * sizeof(float));

    // Normlaize the data --> 0 to 1
    for (int i = 0; i < (*width) * (*height) * (*channels); i++)
    {
        (*data)[i] = (float)udata[i] / 255.0f;
    }

    if (*data == NULL)
    {
        printf("Error loading image.\n of name %s", filename);
        exit(1);
    }
    // Free the loaded image
    stbi_image_free(udata);

    printf("Image size: %d x %d x %d\n", *width, *height,*channels);
}


void write_image(const char *folder_name, char *name, float *data, int width, int height, int channels)
{
    // Create the output file path
    std::string folder(folder_name);
    std::string path = folder + "/" + (std::string)name;
 
    printf("Writing image to %s\n", path.c_str());
    
    // Allocate memory for unsigned char data
    unsigned char *unsigned_char_data = new unsigned char[width * height * channels];

    // Convert from float to unsigned char
    for (int j = 0; j < width * height * channels; ++j)
    {
        //Clipping 
        unsigned_char_data[j] = static_cast<unsigned char>(255.0f * std::max(0.0f, std::min(1.0f, data[j]))); // Clamp values to [0, 1] range
    }

    // Write the image to disk
    if (!stbi_write_png(path.c_str(), width, height, channels, unsigned_char_data, width * channels))
    {
        printf("Failed to write image to %s\n", path.c_str());
    }
    else
    {
        printf("Sucessfully written to %s\n", path.c_str());
    }

    // Free the allocated memory
    delete[] unsigned_char_data;
}

// Device Kernels
__global__ void BatchConvolution(float *image, float *output_image, int width, int height, int channels, int filter_dim)
{
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;

    // Boundary Cond
    if (outRow < height && outCol < width)
    {
        float sum = 0;
        // Looping over mask :D
        for (int filterRow = 0; filterRow < filter_dim; filterRow++)
        {
            for (int filterCol = 0; filterCol < filter_dim; filterCol++)
            {
                int inRow = outRow - filter_dim / 2 + filterRow; // outRow - FilterRaduis + filterRow
                int inCol = outCol - filter_dim / 2 + filterCol; // outCol - FilterRaduis + filterCol

                // Apply boundary conditions (ghost cells)
                inRow = max(0, min(inRow, height - 1));
                inCol = max(0, min(inCol, width - 1));


                // // Check if out of Bounday --> This is useless in case of padding
                // if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                // {
                    for (int c = 0; c < 3; c++)
                    {
                        // Every Channel
                        sum += filter_c[filterRow * filter_dim + filterCol] * (float)image[(inRow * width + inCol) * IMAGE_CHANNELS + c];
                    }
                // }
            }
        }
        output_image[outRow * width + outCol] = sum;
    }

}


int main(int argc, char *argv[])
{

    printf("Hello World\n");

    // Input Arguments
    char *input_dir = argv[1];
    char *output_dir = argv[2];
    char *filter_pth = argv[3];

    // 1. Reading Filter
    int filter_dim;
    float *filter = read_filter(filter_pth,filter_dim);


    // 2. Copy Filter to Constant Memory
    cudaMemcpyToSymbol(filter_c, filter, filter_dim * filter_dim * sizeof(float));

    // 3. Process Images
    // Open the input directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_dir)) != NULL)
    {
        printf("Reading Images from Directory: %s\n", input_dir);

        // Iterate over each file in the directory
        while ((ent = readdir(dir)) != NULL)
        {
            // Filter out directories and special entries
            if (ent->d_type == DT_REG)
            {
                // Step(1) Read Image
                // Concatenate directory path and filename
                char file_path[256];
                snprintf(file_path, sizeof(file_path), "%s/%s", input_dir, ent->d_name);

                printf("Reading Image: %s\n", file_path);

                float *image_data;
                int width, height, channels;
                // 3.1 Host memory allocation & Read Image and 
                read_image(file_path, &image_data, &width, &height, &channels);

                // 3.2 Device Memory Allocation for input                
                float *d_image;
                cudaMalloc((void **)&d_image, sizeof(float) * height * width * channels);

                // 3.3 Transfer input data to device memory
                cudaMemcpy(d_image, image_data, sizeof(float) * height * width * channels, cudaMemcpyHostToDevice);


                // 3.4 Host Memory Allocation for output
                float *output = (float *)malloc(sizeof(float) * height * width);
                
                // 3.5 Device Memory Allocation for output
                float *d_output; // Device pointer for the 2D array
                cudaMalloc((void **)&d_output, sizeof(float) * height * width);

                // for (int i = 0; i < 2; i++)
                // {
                //     for (int j = 0; j < width; j++)
                //     {
                //         printf("%f ", image_data[i * width + j]);
                //     }
                //     printf("\n");
                // }

                // 
               

           

                // 3.6 Convolution
                // Block Size
                dim3 threadsPerBlock(16, 16);
                // Grid Size
                dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
                // Call the kernel
                BatchConvolution<<<numBlocks, threadsPerBlock>>>(d_image, d_output, width, height, IMAGE_CHANNELS, filter_dim);

                // If Errro occurs in Kernel Execution Show it using cudaDeviceSynchronize,cudaGetLastError:D
                cudaDeviceSynchronize();
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess)
                {
                    // in red
                    printf("\033[1;31m");
                    printf("CUDA error: %s\n", cudaGetErrorString(error));
                    // reset color
                    printf("\033[0m");
                }
                
                // 3.7 Transfer output data back to host memory
                cudaMemcpy(output, d_output, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

                // 3.8 Save Image
                // Concatenate directory path and filename
                char out_file_path[256];
                snprintf(out_file_path, sizeof(out_file_path), "%s/%s", output_dir, ent->d_name);
                write_image(output_dir, ent->d_name, output, width, height, 1);


                //         // Verifcation
                //         // Perform convolution
                //         for (int i = 0; i < height; i++) {
                //             for (int j = 0; j < height; j++) {
                //                 float sum =0; // Initialize output at position (i,j) to zero
                //                 // Apply filter
                //                 for (int k = 0; k < FILTER_DIM; k++) {
                //                     for (int l = 0; l < FILTER_DIM; l++) {
                //                         int ni = i - FILTER_DIM / 2 + k;
                //                         int nj = j - FILTER_DIM / 2 + l;
                //                         for (int c = 0; c < IMAGE_CHANNELS; ++c) {
                //                             // Check boundaries
                //                             if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
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

                // 3.9 Free Host Memory
                free(image_data);
                free(output);

                // 3.10 Free Device Memory
                cudaFree(d_image);
                cudaFree(d_output);
            }
        }

        // Close the directory
        closedir(dir);
    }
    else
    {
        // Failed to open directory
        perror("Failed to open Input directory");
        return EXIT_FAILURE;
    }

    // Free memory allocated for the filter in host memory
    free(filter);

    // Free memory allocated for the filter in constant memory
    cudaFree(filter_c);

    // // Images as Batches
    // for(int batch_idx=0;batch_idx<batch_size;batch_idx++){
    //     // 2. Reading Image
    //     // float* image = read_image(file_path){

    //     // }

    //     // read_image(input_folder_pth);
    // }
}