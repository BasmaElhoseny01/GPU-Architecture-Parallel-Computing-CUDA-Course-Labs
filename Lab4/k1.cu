// nvcc -o out_1  ./k1.cu
// ./out_1 ./input ./output ./filter.txt
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
// __constant__ float *filter_c;
__constant__ float filter_c[20 * 20];
// __constant__ float filter_c[FILTER_DIM][FILTER_DIM];

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

// __host__ float *read_image(const char *file_path)
// {
//     printf("Reading image at %s\n", file_path);

//     int width, height, channels;
//     // float *image = stbi_loadf(file_path, &width, &height, &channels, 0); // Each group of three consecutive elements in the array represents the RGB values of one pixel.
//     unsigned char *image = stbi_load(file_path, &width, &height, &channels, 0); // Each group of three consecutive elements in the array represents the RGB values of one pixel.
//     if (image == NULL)
//     {
//         printf("Error: Unable to open file %s\n", file_path);
//         exit(1);
//     }

//     printf("Image size: %d x %d\n", width, height);
//     printf("Channels: %d\n", channels);

//     // assert (width == width);
//     // assert (height == height);
//     // assert (channels == IMAGE_CHANNELS);

//     return image;
// }
void read_image(const char *filename, float **data, int *width, int *height, int *comp)
{
    unsigned char *udata = stbi_load(filename, width, height, comp, 0);
    // convert data to float
    *data = (float *)malloc((*width) * (*height) * (*comp) * sizeof(float));

    for (int i = 0; i < (*width) * (*height) * (*comp); i++)
    {
        (*data)[i] = (float)udata[i] / 255.0f;
    }

    if (*data == NULL)
    {
        printf("Error loading image.\n of name %s", filename);
        exit(1);
    }
    stbi_image_free(udata);
    printf("Image loaded: width = %d, height = %d, comp = %d\n", *width, *height, *comp);
}
// void writeImageToFile(const char *filename, const float *imageData, int width, int height)
// {
//     std::ofstream file(filename, std::ios::out | std::ios::binary);
//     if (!file)
//     {
//         std::cerr << "Error: Unable to open file for writing." << std::endl;
//         return;
//     }

//     // print first 2 rows of image
//     for (int i = 0; i < 2; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             std::cout << imageData[i * width + j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     // Write image width and height to file
//     file.write(reinterpret_cast<const char *>(&width), sizeof(int));
//     file.write(reinterpret_cast<const char *>(&height), sizeof(int));

//     // Write image data to file
//     file.write(reinterpret_cast<const char *>(imageData), width * height);

//     file.close();
// }

void write_image(const char *folder_name, char *name, float *data, int width, int height, int channels)
{
    printf("Writing image to %s\n", folder_name);
    // Create the output file path
    std::string folder(folder_name);
    std::string path = folder + "/" + (std::string)name;
    printf("Trying to Writing image to %s\n", path.c_str());
    // Allocate memory for unsigned char data
    unsigned char *ucharData = new unsigned char[width * height * channels];

    // Convert from float to unsigned char
    for (int j = 0; j < width * height * channels; ++j)
    {
        ucharData[j] = static_cast<unsigned char>(255.0f * std::max(0.0f, std::min(1.0f, data[j]))); // Clamp values to [0, 1] range
    }

    // Write the image as a PNG
    // if (!stbi_write_jpg(path.c_str(), width, height, channels, ucharData, 100))
    if (!stbi_write_png(path.c_str(), width, height, channels, ucharData, width * channels))
    {
        printf("Failed to write image to %s\n", path.c_str());
    }
    else
    {
        printf("Image written to %s\n", path.c_str());
    }

    // Free the allocated memory
    delete[] ucharData;
}

// Device Kernels
// __global__ void BatchConvolution(float image[height][width][IMAGE_CHANNELS], float output_image[height][width])
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

                // Check if out of Bounday --> This is useless in case of padding
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        // Every Channel
                        // printf("[%d][%d]%d\n",filterRow,filterCol,filterRow*filter_dim+filterCol);
                        sum += filter_c[filterRow * filter_dim + filterCol] * (float)image[(inRow * width + inCol) * IMAGE_CHANNELS + c];
                        // sum += filter_c[filterRow][filterCol] * (float)image[(inRow * width + inCol) * IMAGE_CHANNELS + c];
                    }
                    // printf("%f.2    ",sum);
                    // printf("%u\n",(unsigned int)sum);
                }
            }
        }
        // printf("%d\n", (int)output_image[outRow][outCol]);
        // output_image[outRow][outCol] = (float)sum;
        output_image[outRow * width + outCol] = sum;
    }

    //    printf("Hello World\n");
    //    printf("%d\n", (int)output_image[outRow][outCol]);
}

// Kernel adjusted for flattened array
// __global__ void BatchConvolution(float *image, float *output_image, int width, int height, int channels)
// {
//     int outCol = blockDim.x * blockIdx.x + threadIdx.x;
//     int outRow = blockDim.y * blockIdx.y + threadIdx.y;

//     if (outRow < height && outCol < width)
//     {
//         for (int c = 0; c < channels; c++)
//         { // Iterate over each color channel
//             float sum = 0.0f;
//             // Apply filter
//             for (int filterRow = -FILTER_DIM / 2; filterRow <= FILTER_DIM / 2; filterRow++)
//             {
//                 for (int filterCol = -FILTER_DIM / 2; filterCol <= FILTER_DIM / 2; filterCol++)
//                 {
//                     int inRow = min(max(outRow + filterRow, 0), height - 1);
//                     int inCol = min(max(outCol + filterCol, 0), width - 1);
//                     int idx = (inRow * width + inCol) * channels + c;
//                     // int filterIdx = (filterRow + FILTER_DIM / 2) * FILTER_DIM + (filterCol + FILTER_DIM / 2);
//                     sum += image[idx] * filter_c[filterRow][filterCol];
//                 }
//             }
//             output_image[(outRow * width + outCol) * channels + c] = min(max(int(sum), 0), 255);
//         }
//     }
// }
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

    // Allocate memory for the filter in constant memory
    // cudaMalloc(&filter_c, filter_dim * filter_dim * sizeof(float));

    // 2. initialize filter in constant memory
    cudaMemcpyToSymbol(filter_c, filter, filter_dim * filter_dim * sizeof(float));

    // Free dynamically allocated memory
    // free(filter);
    printf("Allocated Filter in Constant Memory\n");

    // 3. Ouptut Memory
    // 3.1. Allocate Host

    // Open the input directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_dir)) != NULL)
    {
        printf("Reading Images from Directory: %s\n", input_dir);

        // int index = 0;

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
                // float *image_data = read_image(file_path);
                float *image_data;
                int width, height, channels;
                read_image(file_path, &image_data, &width, &height, &channels);
                // float *output = (float *)malloc(sizeof(float) * height * width);
                float *output = (float *)malloc(sizeof(float) * height * width);

                // 3.1. Allocate Device
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

                // Step(2) Device Memory Allocation
                float *d_image;
                cudaMalloc((void **)&d_image, sizeof(float) * height * width * channels);

                // Step(3) Copy from host to deivce
                cudaMemcpy(d_image, image_data, sizeof(float) * height * width * channels, cudaMemcpyHostToDevice);

                // Step(4) Call Kernel
                // Kernel
                dim3 threadsPerBlock(16, 16); // Example block size, can be adjusted based on your GPU's architecture
                dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
                // Call the kernel
                BatchConvolution<<<numBlocks, threadsPerBlock>>>(d_image, d_output, width, height, IMAGE_CHANNELS, filter_dim);
                // BatchConvolution<<<GridSize, ThreadsPerBlock>>>((float(*)[width][IMAGE_CHANNELS])d_image, (float(*)[width])d_output);
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
                // cudaGetErrorString(cudaGetLastError());
                // print error

                // Step(5) Transfer data back to host memory

                cudaMemcpy(output, d_output, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

                // Save Image
                // Concatenate directory path and filename
                char out_file_path[256];
                snprintf(out_file_path, sizeof(out_file_path), "%s/%s", output_dir, ent->d_name);

                // // print the first 2 rows of the image
                // for (int i = 0; i < 2; i++)
                // {
                //     for (int j = 0; j < width; j++)
                //     {
                //         printf("%f ", output[i * width + j]);
                //     }
                //     printf("\n");
                // }
                // stbi_write_jpg(out_file_path, width, height, 1, output, 90);
                // writeImageToFile(out_file_path, output, width, height);

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

                // free memory
                free(image_data);
                free(output);
                cudaFree(d_image);
                cudaFree(d_output);
            }
        }

        closedir(dir);
    }
    else
    {
        // Failed to open directory
        // Can;t open directory at path
        perror("Failed to open Input directory");
        return EXIT_FAILURE;
    }

    // Free memory allocated for the filter in constant memory
    // cudaFree(filter_c);

    // // Images as Batches
    // for(int batch_idx=0;batch_idx<batch_size;batch_idx++){
    //     // 2. Reading Image
    //     // float* image = read_image(file_path){

    //     // }

    //     // read_image(input_folder_pth);
    // }
}