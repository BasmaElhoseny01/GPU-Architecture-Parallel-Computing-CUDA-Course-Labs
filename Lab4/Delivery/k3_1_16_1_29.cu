// Match Input Tile
// All Threads Participate in the Loading of the Tile but some threads do not participate in the computation of the output tile
// nvcc -o out_3  ./k3.cu
// ./out_3 ./input ./output_k3 2 ./filters/avg_9_9.txt
// nvprof out_3 ./testfile.txt ./out.txt

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

// #define OUTPUT_TILE_DIM 32
#define OUTPUT_TILE_DIM 16
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

__global__ void output_tile_convolution(float *image, float *output_image, int width, int height, int batch_size, int filter_dim, const int INPUT_TILE_DIM, float *filter_c)
{
    // Store all elements needed to compute output in shared memory for the 3 channels
    extern __shared__ float sh_mem[];

    // Number of Pixels Each Thread is responsible for ceil(input tile pixels/output tile pixels)
    int no_pixels_per_thread = (INPUT_TILE_DIM * INPUT_TILE_DIM + OUTPUT_TILE_DIM * OUTPUT_TILE_DIM - 1) / (OUTPUT_TILE_DIM * OUTPUT_TILE_DIM);

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x; // Local To Block

    // Start Index in both directions is from block index in the Grid with back steps by filterdim/2
    int start_index_x = blockIdx.x * OUTPUT_TILE_DIM - (filter_dim / 2);
    int start_index_y = blockIdx.y * OUTPUT_TILE_DIM - (filter_dim / 2);

    int batch_index = blockIdx.z;

    for (int i = 0; i < no_pixels_per_thread; i++)
    {
        int step = blockDim.x * blockDim.y;

        int index_before_linearization = thread_id + i * step;
        int lin_col = start_index_x + index_before_linearization % INPUT_TILE_DIM;
        int lin_row = start_index_y + index_before_linearization / INPUT_TILE_DIM;

        if (lin_row >= 0 && lin_col >= 0 && lin_col < width && lin_row < height)
        {
            for (int c = 0; c < IMAGE_CHANNELS; c++)
            {
                int image_index = (lin_row * width + lin_col +
                                   batch_index * width * height) *
                                      IMAGE_CHANNELS +
                                  c;
                int shared_index = thread_id * IMAGE_CHANNELS + c + i * (blockDim.x * blockDim.y * IMAGE_CHANNELS);

                if (shared_index < INPUT_TILE_DIM * INPUT_TILE_DIM * 3)
                {
                    sh_mem[shared_index] = image[image_index];
                }
            }
        }
        else
        {
            // Out of bounds handling
            for (int c = 0; c < IMAGE_CHANNELS; c++)
            {
                int shared_index = thread_id * IMAGE_CHANNELS + c + i * (blockDim.x * blockDim.y * IMAGE_CHANNELS);

                if (shared_index < INPUT_TILE_DIM * INPUT_TILE_DIM * 3)
                {

                    sh_mem[shared_index] = 0.0; // Assign some default value or handle differently based on your requirement
                }
            }
        }
    }

    __syncthreads();

    // Each Thread is now Responsible for 1 output
    // OutPut Image Indices
    int out_row = blockIdx.y * OUTPUT_TILE_DIM + threadIdx.y;
    int out_col = blockIdx.x * OUTPUT_TILE_DIM + threadIdx.x;

    if (out_row >= 0 && out_row < height && out_col >= 0 && out_col < width)
    {
        //     // printf("[%d,%d]   %d\n",out_row,out_col,out_row * width + out_col);

        float sum = 0.0;
        // Looping over mask :D
        for (int filterRow = 0; filterRow < filter_dim; filterRow++)
        {
            for (int filterCol = 0; filterCol < filter_dim; filterCol++)
            {
                // Ensure indices are within the boundaries of sh_mem
                int sh_row = threadIdx.y + filterRow;
                int sh_col = threadIdx.x + filterCol;

                if (sh_row >= 0 && sh_row < INPUT_TILE_DIM && sh_col >= 0 && sh_col < INPUT_TILE_DIM)
                {
                    // For Every Channel
                    for (int c = 0; c < 3; c++)
                    {
                        if (blockIdx.x == 1 && blockIdx.y == 0)
                        {

                            // printf("sh[%d]\n", (sh_row * blockDim.x + sh_col) * IMAGE_CHANNELS + c);
                        }
                        // Every Channel
                        sum += filter_c[filterRow * filter_dim + filterCol] * sh_mem[(sh_row * INPUT_TILE_DIM + sh_col) * IMAGE_CHANNELS + c];
                        // sum = sh_mem[(sh_row * blockDim.x + sh_col) * IMAGE_CHANNELS];
                    }
                }
            }
        }

        output_image[batch_index * (width * height) + out_row * width + out_col] = sum;
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

    int batch_index = 0;
    if ((dir = opendir(input_dir)) != NULL)
    {
        while (1)
        {
            printf("\n\nBatch [%d] n", batch_index);
            batch_index++;

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
            // dim3 threadsPerBlock(16, 16, 4);
            // // Grid Size
            // dim3 numBlocks((IMAGE_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
            //    (IMAGE_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y,
            //    (batch_counter + threadsPerBlock.z - 1) / threadsPerBlock.z);

            // // Call the kernel
            // BatchConvolution<<<numBlocks, threadsPerBlock>>>(d_batched_images, d_output, IMAGE_WIDTH, IMAGE_HEIGHT, batch_counter, filter_dim);

            // input tile
            int input_tile_dim = OUTPUT_TILE_DIM + filter_dim - 1;

            dim3 threads_per_block(OUTPUT_TILE_DIM, OUTPUT_TILE_DIM, 1);

            dim3 blocks_num(
                (IMAGE_WIDTH + OUTPUT_TILE_DIM - 1) / OUTPUT_TILE_DIM,
                (IMAGE_HEIGHT + OUTPUT_TILE_DIM - 1) / OUTPUT_TILE_DIM,
                (batch_counter + threads_per_block.z - 1) / threads_per_block.z);
            // 1);

            // printf("%d\n", blocks_num.x);
            // printf("%d\n", blocks_num.y);
            // printf("%d\n", input_tile_dim * input_tile_dim * 3);
            // printf("%d\n", input_tile_dim);
            // return 0;

            output_tile_convolution<<<blocks_num, threads_per_block, input_tile_dim * input_tile_dim * 3 * sizeof(float)>>>(d_batched_images, d_output, IMAGE_WIDTH, IMAGE_HEIGHT, batch_counter, filter_dim, input_tile_dim, filter_c);

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