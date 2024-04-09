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
#include "stb\stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb\stb_image_write.h"

#define FILTER_DIM 9
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 960
#define IMAGE_CHANNELS 3

// Declare Constant Memory
__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

// Host Functions
__host__ float *read_filter(const char *file_path)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        printf("Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    int n;
    fscanf(file, "%d", &n);
    printf("Filter size: %d\n", n);

    printf("Filter Applied:\n");
    float *filter = (float *)malloc(n * n * sizeof(float));
    for (int i = 0; i < n * n; i++)
    {
        fscanf(file, "%f", &filter[i]);
        printf("%f ", filter[i]);

        if (i % n == n - 1)
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

//     // assert (width == IMAGE_WIDTH);
//     // assert (height == IMAGE_HEIGHT);
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
void writeImageToFile(const char *filename, const float *imageData, int width, int height)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file)
    {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }

    // print first 2 rows of image
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < width; j++)
        {
            std::cout << imageData[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Write image width and height to file
    file.write(reinterpret_cast<const char *>(&width), sizeof(int));
    file.write(reinterpret_cast<const char *>(&height), sizeof(int));

    // Write image data to file
    file.write(reinterpret_cast<const char *>(imageData), width * height);

    file.close();
}

// Device Kernels
// __global__ void BatchConvolution(float image[IMAGE_HEIGHT][IMAGE_WIDTH][IMAGE_CHANNELS], float output_image[IMAGE_HEIGHT][IMAGE_WIDTH])
// {
//     int outRow = blockDim.y * blockIdx.y + threadIdx.y;
//     int outCol = blockDim.x * blockIdx.x + threadIdx.x;

//     // Boundary Cond
//     if (outRow < IMAGE_HEIGHT && outCol < IMAGE_WIDTH)
//     {
//         float sum = 0;
//         // Looping over mask :D
//         for (int filterRow = 0; filterRow < FILTER_DIM; filterRow++)
//         {
//             for (int filterCol = 0; filterCol < FILTER_DIM; filterCol++)
//             {
//                 int inRow = outRow - FILTER_DIM / 2 + filterRow; // outRow - FilterRaduis + filterRow
//                 int inCol = outCol - FILTER_DIM / 2 + filterCol; // outCol - FilterRaduis + filterCol

//                 // Check if out of Bounday --> This is useless in case of padding
//                 if (inRow >= 0 && inRow < IMAGE_HEIGHT && inCol >= 0 && inCol < IMAGE_WIDTH)
//                 {
//                     for (int c = 0; c < 3; c++)
//                     {
//                         // Every Channel
//                         sum += filter_c[filterRow][filterCol] * (float)image[inRow][inCol][c];
//                     }
//                     // printf("%f.2    ",sum);
//                     // printf("%u\n",(unsigned int)sum);
//                 }
//             }
//         }
//         // printf("%d\n", (int)output_image[outRow][outCol]);
//         output_image[outRow][outCol] = (float)sum;
//     }

//     //    printf("Hello World\n");
//     //    printf("%d\n", (int)output_image[outRow][outCol]);
// }

// Kernel adjusted for flattened array
__global__ void BatchConvolution(float *image, float *output_image, int width, int height, int channels)
{
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;

    if (outRow < height && outCol < width)
    {
        for (int c = 0; c < channels; c++)
        { // Iterate over each color channel
            float sum = 0.0f;
            // Apply filter
            for (int filterRow = -FILTER_DIM / 2; filterRow <= FILTER_DIM / 2; filterRow++)
            {
                for (int filterCol = -FILTER_DIM / 2; filterCol <= FILTER_DIM / 2; filterCol++)
                {
                    int inRow = min(max(outRow + filterRow, 0), height - 1);
                    int inCol = min(max(outCol + filterCol, 0), width - 1);
                    int idx = (inRow * width + inCol) * channels + c;
                    // int filterIdx = (filterRow + FILTER_DIM / 2) * FILTER_DIM + (filterCol + FILTER_DIM / 2);
                    sum += image[idx] * filter_c[filterRow][filterCol];
                }
            }
            output_image[(outRow * width + outCol) * channels + c] = min(max(int(sum), 0), 255);
        }
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
    float *filter = read_filter(filter_pth);

    // 2. initialize filter in constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM * FILTER_DIM * sizeof(float));
    printf("Allocated Filter in Constant Memory\n");

    // 3. Ouptut Memory
    // 3.1. Allocate Host
    float *output = (float *)malloc(sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH);

    // 3.1. Allocate Device
    float *d_output; // Device pointer for the 2D array
    cudaMalloc((void **)&d_output, sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH);

    // Open the input directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_dir)) != NULL)
    {
        printf("Reading Images from Directory: %s\n", input_dir);

        int index = 0;

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

                // for (int i = 0; i < 2; i++)
                // {
                //     for (int j = 0; j < IMAGE_WIDTH; j++)
                //     {
                //         printf("%f ", image_data[i * IMAGE_WIDTH + j]);
                //     }
                //     printf("\n");
                // }

                // Step(2) Device Memory Allocation
                float *d_image;
                cudaMalloc((void **)&d_image, sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS);

                // Step(3) Copy from host to deivce
                cudaMemcpy(d_image, image_data, sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS, cudaMemcpyHostToDevice);

                // Step(4) Call Kernel
                // Kernel
                // dim3 ThreadsPerBlock(16, 16);
                // dim3 GridSize((IMAGE_WIDTH - 1) / ThreadsPerBlock.x + 1, (IMAGE_HEIGHT - 1) / ThreadsPerBlock.y + 1);
                dim3 threadsPerBlock(16, 16); // Example block size, can be adjusted based on your GPU's architecture
                dim3 numBlocks((IMAGE_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                               (IMAGE_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
                // Call the kernel
                BatchConvolution<<<numBlocks, threadsPerBlock>>>(d_image, d_output, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS);
                // BatchConvolution<<<GridSize, ThreadsPerBlock>>>((float(*)[IMAGE_WIDTH][IMAGE_CHANNELS])d_image, (float(*)[IMAGE_WIDTH])d_output);
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
                cudaMemcpy(output, d_output, sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH, cudaMemcpyDeviceToHost);

                // Save Image
                // Concatenate directory path and filename
                char out_file_path[256];
                snprintf(out_file_path, sizeof(out_file_path), "%s/%s", output_dir, ent->d_name);

                // print the first 2 rows of the image

                stbi_write_jpg(out_file_path, IMAGE_WIDTH, IMAGE_HEIGHT, 1, output, 90);

                //         // Verifcation
                //         // Perform convolution
                //         for (int i = 0; i < IMAGE_HEIGHT; i++) {
                //             for (int j = 0; j < IMAGE_HEIGHT; j++) {
                //                 float sum =0; // Initialize output at position (i,j) to zero
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
    else
    {
        // Failed to open directory
        // Can;t open directory at path
        perror("Failed to open Input directory");
        return EXIT_FAILURE;
    }

    // // Images as Batches
    // for(int batch_idx=0;batch_idx<batch_size;batch_idx++){
    //     // 2. Reading Image
    //     // float* image = read_image(file_path){

    //     // }

    //     // read_image(input_folder_pth);
    // }
}