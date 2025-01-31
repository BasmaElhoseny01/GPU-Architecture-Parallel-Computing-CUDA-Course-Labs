// nvcc -o out_1  ./k1.cu
// ./out_1 ./input ./output_k1 2 ./filters/avg_9_9.txt
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
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define IMAGE_CHANNELS 3

// Declare Constant Memory
// Max is 400 floating element :D
// __constant__ float filter_c[20 * 20];

// Host Functions
__host__ float *read_filter(const char *file_path, int &filter_dim)
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

// Function to get the dimensions of the first image in the directory
__host__ void get_dimensions(const char *input_dir, int *width, int *height, int *channels)
{
    DIR *dir;
    struct dirent *ent;

    // Open the directory
    if ((dir = opendir(input_dir)) != NULL)
    {
        // Iterate over each file in the directory
        while ((ent = readdir(dir)) != NULL)
        {
            // Filter out directories and special entries
            if (ent->d_type == DT_REG)
            {
                // Concatenate directory path and filename
                char file_path[256];
                snprintf(file_path, sizeof(file_path), "%s/%s", input_dir, ent->d_name);

                // Load the image using stb_image.h
                int w, h, c;
                unsigned char *image_data = stbi_load(file_path, &w, &h, &c, 0);

                if (image_data != NULL)
                {
                    // Assign dimensions
                    *width = w;
                    *height = h;
                    *channels = c;

                    printf("[All]Image Size: %d x %d x %d\n", *width, *height, *channels);

                    // Free the image data
                    stbi_image_free(image_data);

                    // Close the directory and return
                    closedir(dir);
                    return;
                }
                else
                {
                    fprintf(stderr, "Error loading image: %s\n", file_path);
                }
            }
        }
        // Close the directory
        closedir(dir);
    }
    else
    {
        // Error opening directory
        perror("Unable to open directory");
    }
}

__host__ void load_input_image(const char *folder_name, char *file_name, float **image, int *width, int *height, int *channels)
{
    // Concatenate directory path and filename
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", folder_name, file_name);

    // Read Image
    unsigned char *image_data = stbi_load(path, width, height, channels, 0);

    // Host Memory Allocation & convert data from unsigned char to float
    *image = (float *)malloc(sizeof(float) * (*width) * (*height) * (*channels));

    // Normlaization
    for (int i = 0; i < (*height) * (*width) * (*channels); i++)
    {
        (*image)[i] = (float)image_data[i] / 255.0f;
    }

    if (*image == NULL)
    {
        printf("Error loading image\n");
        exit(1);
    }

    // Free the loaded image
    stbi_image_free(image_data);
}

__host__ void dump_output_image(const char *out_directory, char *file_name, float *output_img, int width, int height, int channels)
{

    std::string directory(out_directory);
    std::string path = directory + "/" + (std::string)file_name;

    // Memory allocation
    unsigned char *unsigned_char_data = new unsigned char[width * height * channels];

    for (int j = 0; j < height * width * channels; ++j)
    {
        // Clipping
        unsigned_char_data[j] = static_cast<unsigned char>(255.0f * std::max(0.0f, std::min(1.0f, output_img[j]))); // Clamp values to [0, 1] range
    }

    bool result = stbi_write_png(path.c_str(), width, height, channels, unsigned_char_data, width * channels);

    if (!result)
    {
        printf("Failed to write image\n");
    }

    // Free memory
    delete[] unsigned_char_data;
}

// Device Kernels
__global__ void BatchConvolution(float *image, float *output_image, int width, int height, int batch_size, int filter_dim, float *filter_c)
{
    // Each Thread is responsible for one pixel in the output image
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    int outDepth = blockDim.z * blockIdx.z + threadIdx.z;

    // Boundary Cond
    if (outRow < height && outCol < width && outDepth < batch_size)
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
                        // Every Channel
                        sum += filter_c[filterRow * filter_dim + filterCol] * (float)image[(outDepth * height * width + inRow * width + inCol) * IMAGE_CHANNELS + c];
                }
                // else
                //{
                //     // ghost Cells
                //     sum += 0.0;
                // }
            }
        }
        output_image[(outDepth * height * width) + (outRow * width) + outCol] = sum;
    }
}

int main(int argc, char *argv[])
{

    printf("Hello World\n");

    // Input Arguments
    char *input_dir = argv[1];
    char *output_dir = argv[2];
    int batch_size = atoi(argv[3]);
    char *filter_pth = argv[4];

    // 1. Reading Filter
    int filter_dim;
    float *filter = read_filter(filter_pth, filter_dim);
    // 1.1. Copy Filter to cuda memory
    float *filter_c;
    cudaMalloc((void **)&filter_c, sizeof(float) * filter_dim * filter_dim);
    cudaMemcpy(filter_c, filter, sizeof(float) * filter_dim * filter_dim, cudaMemcpyHostToDevice);

    // 2. Copy Filter to Constant Memory
    // cudaMemcpyToSymbol(filter_c, filter, filter_dim * filter_dim * sizeof(float));

    // 3. Process Images
    printf("Reading Images from Directory: %s\n", input_dir);

    // 3.1. Get Images Dimensions
    int IMAGE_WIDTH, IMAGE_HEIGHT, image_channels;
    get_dimensions(input_dir, &IMAGE_WIDTH, &IMAGE_HEIGHT, &image_channels);

    // 3.2. Host Memory Allocations
    // Allocate memory to store filenames for each image in the batch
    char **image_filenames = (char **)malloc(batch_size * sizeof(char *));
    // Allocate memory for each individual filename string
    for (int i = 0; i < batch_size; i++)
    {
        image_filenames[i] = (char *)malloc(256 * sizeof(char)); // Assuming max filename length is 256
    }

    // 3.3. Device Memory Allocations
    // Allocate device memory for batched input
    float *d_batched_images;
    cudaMalloc((void **)&d_batched_images, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * batch_size);

    // 4. Process Images in Batches
    DIR *dir;
    struct dirent *ent;

    int batch_idx = 0;
    if ((dir = opendir(input_dir)) != NULL)
    {
        while (1)
        {
            printf("\n\nBatch [%d] n", batch_idx);
            batch_idx++;

            // Counter for images in batch
            int batch_counter = 0;

            // Step(1) Read Images Batches
            while (batch_counter < batch_size)
            {
                if ((ent = readdir(dir)) == NULL)
                {
                    // No more files
                    break;
                }

                // Filter out directories and special entries
                if (ent->d_type == DT_REG)
                {

                    // Read Image
                    snprintf(image_filenames[batch_counter], 256, "%s", ent->d_name);

                    float *image_data;
                    int width, height, channels;
                    // Host memory allocation & Read Image and
                    load_input_image(input_dir, ent->d_name, &image_data, &width, &height, &channels);

                    // Check on input image dimensions
                    assert(width == IMAGE_WIDTH && height == IMAGE_HEIGHT);

                    // Transfer input data to device memory
                    cudaMemcpy(d_batched_images + batch_counter * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS, image_data, sizeof(float) * height * width * IMAGE_CHANNELS, cudaMemcpyHostToDevice);

                    // Free host memory for image data
                    free(image_data);

                    // Increment Batch Counter
                    batch_counter++;
                }
            }

            if (batch_counter == 0)
            {
                // Empty Batch
                break;
            }

            // Step(2) Process Batch
            printf("Prepocessing Batch ...\n");

            // Host Memory Allocation for output
            float *output = (float *)malloc(sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT * batch_counter);

            // Device Memory Allocation for output
            float *d_output; // Device pointer for the 2D array
            cudaMalloc((void **)&d_output, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT * batch_counter);

            // Convolution
            // Block Size
            dim3 threadsPerBlock(16, 16, 1);

            // Grid Size
            dim3 numBlocks((IMAGE_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (IMAGE_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (batch_counter + threadsPerBlock.z - 1) / threadsPerBlock.z);

            printf("batch counter %i\n", (batch_counter + threadsPerBlock.z - 1) / threadsPerBlock.z);

            // Call the kernel
            BatchConvolution<<<numBlocks, threadsPerBlock>>>(d_batched_images, d_output, IMAGE_WIDTH, IMAGE_HEIGHT, batch_counter, filter_dim, filter_c);

            // If Error occurs in Kernel Execution Show it using cudaDeviceSynchronize,cudaGetLastError:D
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

            // Transfer output data back to host memory
            cudaMemcpy(output, d_output, sizeof(float) * IMAGE_HEIGHT * IMAGE_WIDTH * batch_counter, cudaMemcpyDeviceToHost);

            // Free Device Memory
            cudaFree(d_output);

            // Step(3) Save output images Batch
            printf("Writing Batch ...\n");
            for (int i = 0; i < batch_counter; i++)
            {
                // 3.8 Save Image
                // Concatenate directory path and filename
                dump_output_image(output_dir, image_filenames[i], output + (i * IMAGE_HEIGHT * IMAGE_WIDTH), IMAGE_WIDTH, IMAGE_HEIGHT, 1);
            }

            // Free Host Memory
            free(output);

            if (batch_counter < batch_size)
            {
                break;
            }
        }

        // Free Host Memory
        // Free memory allocated for the filter in host memory
        free(filter);
        // Free memory allocated for the filter in host memory
        free(image_filenames);

        // Free Device Memory
        // Free memory allocated for the filter in constant memory
        cudaFree(filter_c);
        // Free memory allocated for the batched images in device shared memory
        cudaFree(d_batched_images);

        // Close the diresctory
        closedir(dir);
    }
    else
    {
        // Error opening directory
        perror("Unable to open directory");
    }
}