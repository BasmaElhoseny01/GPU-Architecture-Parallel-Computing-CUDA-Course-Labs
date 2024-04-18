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
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_CHANNELS 3

#define OUTPUT_TILE_DIM 7
// Declare Constant Memory
// Max is 400 floating element :D
__constant__ float filter_c[20 * 20];

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

// __host__ void read_images_batch
__host__ void read_image(const char *folder_name, char *file_name, float **data, int *width, int *height, int *channels)
{
    // Concatenate directory path and filename
    char file_path[256];
    snprintf(file_path, sizeof(file_path), "%s/%s", folder_name, file_name);

    printf("Reading Image: %s\n", file_path);

    // Read Image
    unsigned char *udata = stbi_load(file_path, width, height, channels, 0);

    // Host Memory Allocation & convert data from unsigned char to float
    *data = (float *)malloc((*width) * (*height) * (*channels) * sizeof(float));

    // Normlaize the data --> 0 to 1
    for (int i = 0; i < (*width) * (*height) * (*channels); i++)
    {
        (*data)[i] = (float)udata[i] / 255.0f;
    }

    if (*data == NULL)
    {
        printf("Error loading image at %s", file_path);
        exit(1);
    }
    // Free the loaded image
    stbi_image_free(udata);

    printf("Image size: %d x %d x %d\n", *width, *height, *channels);
}

__host__ void write_image(const char *folder_name, char *file_name, float *data, int width, int height, int channels)
{
    // Create the output file path
    std::string folder(folder_name);
    std::string path = folder + "/" + (std::string)file_name;

    printf("Writing image to %s\n", path.c_str());

    // Allocate memory for unsigned char data
    unsigned char *unsigned_char_data = new unsigned char[width * height * channels];

    // Convert from float to unsigned char
    for (int j = 0; j < width * height * channels; ++j)
    {
        // Clipping
        unsigned_char_data[j] = static_cast<unsigned char>(255.0f * std::max(0.0f, std::min(1.0f, data[j]))); // Clamp values to [0, 1] range
    }

    // Write the image to disk
    if (!stbi_write_png(path.c_str(), width, height, channels, unsigned_char_data, width * channels))
    {
        printf("Failed to write image to %s\n", path.c_str());
    }

    // Free the allocated memory
    delete[] unsigned_char_data;
}

__global__ void output_tile_convolution(float *image, float *output_image, int width, int height, int batch_size, int filter_dim, const int INPUT_TILE_DIM)
{
    // Store all elements needed to compute output in shared memory for the 3 channels
    extern __shared__ float sh_mem[];

    //  No of Pixels Each Thread is responsible for ceil(input tile pixels/outputtile pixels)
    int no_pixles_per_thread_x = (INPUT_TILE_DIM + OUTPUT_TILE_DIM - 1) / (OUTPUT_TILE_DIM);
    int no_pixles_per_thread_y = (INPUT_TILE_DIM + OUTPUT_TILE_DIM - 1) / (OUTPUT_TILE_DIM);

    // printf("no_pixles_per_thread_x %d", no_pixles_per_thread_x); // 2
    // printf("no_pixles_per_thread_y %d", no_pixles_per_thread_y); // 2
    // printf("no_pixles_per_thread_y %d", no_pixles_per_thread_y * no_pixles_per_thread_x); // 3

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // printf("thread_id %d\n",thread_id);  // 0-8 :D

    // int start_index_x = blockIdx.x * OUTPUT_TILE_DIM - filter_dim / 2;
    // int start_index_y = blockIdx.y * OUTPUT_TILE_DIM - filter_dim / 2;

    // printf("start_index_x %d\n", start_index_x); //-1
    // printf("start_index_y %d\n", start_index_y); //-1

    int load_counter = 0;
    // printf("%d width\n", width);
    // printf("%d filter_dim\n,", filter_dim);
    // printf("%d width + filter_dim / 2\n", width + filter_dim / 2);
    // printf("%d height\n",height );
    // printf("%d height + filter_dim / 2\n",height + filter_dim / 2);

    int out_row_temp = blockIdx.y * OUTPUT_TILE_DIM + threadIdx.y;
    int out_col_temp = blockIdx.x * OUTPUT_TILE_DIM + threadIdx.x;

    for (int start_index_y = blockIdx.y * OUTPUT_TILE_DIM - filter_dim / 2; start_index_y < (height + filter_dim / 2); start_index_y += OUTPUT_TILE_DIM)
    {
        // if (blockIdx.x == 0 && blockIdx.y == 0 && thread_id == 0)
        // {
        //     printf("start_index_x %d\n", start_index_x); //-1
        //     // printf("start_index_y %d\n", start_index_y); //-1
        // }
        for (int start_index_x = blockIdx.x * OUTPUT_TILE_DIM - filter_dim / 2; start_index_x < (width + filter_dim / 2); start_index_x += OUTPUT_TILE_DIM)

        {

            // if (blockIdx.x == 0 && blockIdx.y == 0 && thread_id == 0)
            // {
            // printf("start_index_x %d,start_index_y %d\n", start_index_x,start_index_y); //-1
            // printf("start_index_y %d\n", start_index_y); //-1
            // }

            // if (out_row_temp >= 0 && out_col_temp >= 0 && out_col_temp < width && out_row_temp < height)
            // if (((start_index_x + threadIdx.x) >= 0) && ((start_index_y + threadIdx.y) >= 0))
            // int x=start_index_x + (int)threadIdx.x;
            if ((start_index_x + (int)threadIdx.x)>=0 && (start_index_y + (int)threadIdx.y)>=0 )
            // if ((start_index_x + threadIdx.x) >= 0 && (start_index_y + threadIdx.y) >= 0 && (start_index_x + threadIdx.x) < width && (start_index_y + threadIdx.y) < height)
            {
                for (int c = 0; c < IMAGE_CHANNELS; c++)
                {
                    // if (blockIdx.x == 0 && blockIdx.y == 0 && (threadIdx.x==0 && threadIdx.y==0))
                    if (blockIdx.x == 0 && blockIdx.y == 0)
                    // if (blockIdx.x == 0 && blockIdx.y == 0 &&((threadIdx.y ==0 && threadIdx.x==1) ||(threadIdx.y ==0 && threadIdx.x==0)))
                    {
                        // printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS+c);
                        // printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) );
                        // printf("%d write %d \n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c +load_counter * blockDim.x * blockDim.y ,(start_index_y * OUTPUT_TILE_DIM + start_index_x + thread_id) * IMAGE_CHANNELS + c);

                        // printf("SHM [%d]  THID[%d]\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c +load_counter * blockDim.x * blockDim.y,thread_id);
                        // printf("SHM [%d] IN[%d]  THID[%d] [%d,%d]\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * (blockDim.x * blockDim.y * IMAGE_CHANNELS), ((start_index_y + threadIdx.y) * (width + filter_dim / 2 + 1) + (start_index_x + threadIdx.x)) * /*IMAGE_CHANNELS + c*/ 1, thread_id, start_index_x, start_index_y);
                        // printf("(start_index_x + threadIdx.x):%d %d (start_index_y + threadIdx.y):%d %d \n",(start_index_x + threadIdx.x),(start_index_x + threadIdx.x)>=0,(start_index_y + threadIdx.y),(start_index_y + threadIdx.y)>=0);
                        // printf("(start_index_x + threadIdx.x):%d\n",(start_index_x + threadIdx.x));
                        // printf("start_index_x: %d , threadIdx.x %d , total(%d)\n",start_index_x,threadIdx.x,x);
                       
                        printf("SHM [%d] IN[%d,%d] [%d] THID[%d] [%d,%d]\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * (blockDim.x * blockDim.y * IMAGE_CHANNELS), (start_index_y + threadIdx.y) , (start_index_x + threadIdx.x),((start_index_y + threadIdx.y) * (width + filter_dim / 2 + 1) + (start_index_x + threadIdx.x))*IMAGE_CHANNELS + c, thread_id, start_index_x, start_index_y);

                        // printf("%d write %d \n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c +load_counter * blockDim.x * blockDim.y ,(start_index_y * OUTPUT_TILE_DIM + start_index_x + thread_id) * IMAGE_CHANNELS + c);
                        // printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) +load_counter * blockDim.x * blockDim.y );
                    }

                    // if ((threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y < INPUT_TILE_DIM * INPUT_TILE_DIM)
                    // {
                    //     // For Threads at end has nothing else to load

                    //     sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS +c + load_counter * (blockDim.x * blockDim.y*IMAGE_CHANNELS)] = image[(start_index_y * OUTPUT_TILE_DIM + start_index_x + thread_id) * IMAGE_CHANNELS + c];
                    //     // printf("%d\n",(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y);
                    //     // if (blockIdx.x == 0 && blockIdx.y == 0)
                    //     // {

                    //     //     printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c);
                    //     // }
                    // }
                }
            }
            else
            {
                // Ghost Padding
                for (int c = 0; c < IMAGE_CHANNELS; c++)
                {
                    // if (blockIdx.x == 0 && blockIdx.y == 0 && (threadIdx.x==0 && threadIdx.y==0))
                    if (blockIdx.x == 0 && blockIdx.y == 0)
                    {
                        printf("SHM [%d] IN[%d,%d] gh:%d THID[%d] [%d,%d]\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * (blockDim.x * blockDim.y * IMAGE_CHANNELS), (start_index_y + threadIdx.y) , (start_index_x + threadIdx.x),0, thread_id, start_index_x, start_index_y);

                        // printf("SHM [%d] %d THID[%d]\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * (blockDim.x * blockDim.y * IMAGE_CHANNELS), 0, thread_id);
                        // printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) + load_counter * blockDim.x * blockDim.y);
                        // printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS +c);
                        // printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) );
                    }

                    // if ((threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y < INPUT_TILE_DIM * INPUT_TILE_DIM)
                    // {
                    //     // For Threads at end has nothing else to load
                    //     sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS +c + load_counter * (blockDim.x * blockDim.y*IMAGE_CHANNELS)] = 0.0;
                    //     // printf("%d\n",(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y);
                    //     // printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + load_counter * blockDim.x * blockDim.y);
                    //     // if (blockIdx.x == 0 && blockIdx.y == 0)
                    //     // {

                    //     //     printf("%d\n", (threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c);
                    //     // }
                    // }
                }
            }

            load_counter += 1;
        }
    }
    // if (max_x > 512)
    // {
    //     printf(
    //         "Error\n\n");
    // }

    // if (max_y > 256)
    // {
    //     printf(
    //         "Error\n\n");
    // }
    // printf("max_y\n",max_y);

    // for (int i = 0; i < no_pixles_per_thread_x * no_pixles_per_thread_y; i++)
    // {
    //     // Add Thread to the start index :D
    //     // printf("start_index_x %d\n", start_index_x);
    //     // printf("start_index_y %d\n", start_index_y);

    //     if ((start_index_x + threadIdx.x) >= 0 && (start_index_y + threadIdx.y) >= 0 && (start_index_x + threadIdx.x) < width && (start_index_y + threadIdx.y) < height)
    //     {
    //         for (int c = 0; c < IMAGE_CHANNELS; c++)
    //         {
    //             if ((threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + i * blockDim.x * blockDim.y < INPUT_TILE_DIM * INPUT_TILE_DIM)
    //             {
    //                 // Boundary Condition on SHM
    //                 sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + i * blockDim.x * blockDim.y] = image[(start_index_y * OUTPUT_TILE_DIM + start_index_x + thread_id) * IMAGE_CHANNELS + c];
    //             }
    //             // sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + i * blockDim.x * blockDim.y] = 0;
    //             // if (c == 0)
    //             // {

    //             //     // printf("%d\n", (start_index_y * OUTPUT_TILE_DIM + start_index_x + thread_id) * IMAGE_CHANNELS + c); // 0-27 :D
    //             // }
    //         }
    //     }
    //     else
    //     {
    //         // Ghost Padding
    //         for (int c = 0; c < IMAGE_CHANNELS; c++)
    //         {
    //             if ((threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + i * blockDim.x * blockDim.y < INPUT_TILE_DIM * INPUT_TILE_DIM)
    //             {
    //                 // Boundary Condition on SHM
    //                 sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c + i * blockDim.x * blockDim.y] = 0.0;
    //                 // printf("Will break\n");
    //                 // printf("%d,%d,%i\n",threadIdx.x,threadIdx.y,i);
    //             }
    //         }
    //     }

    //     // Move Filter :D
    //     // Move Right
    //     if ((start_index_x + filter_dim) >= width)
    //     {
    //         // Reset start_index_x :D & Move Down
    //         start_index_x = blockIdx.x * OUTPUT_TILE_DIM - filter_dim / 2;
    //         start_index_y += filter_dim;
    //     }
    //     else
    //     {
    //         // Move Right
    //         start_index_x += filter_dim;
    //     }
    // }
    __syncthreads();

    // Each Thread is now Responsible for 1 output
    // OutPut Image Indices
    int out_row = blockIdx.y * OUTPUT_TILE_DIM + threadIdx.y;
    int out_col = blockIdx.x * OUTPUT_TILE_DIM + threadIdx.x;

    if (out_row >= 0 && out_row < height && out_col >= 0 && out_col < width)
    {

        float sum = 0.0;
        // Looping over mask :D
        for (int filterRow = 0; filterRow < filter_dim; filterRow++)
        {
            for (int filterCol = 0; filterCol < filter_dim; filterCol++)
            {
                // For Every Channel
                for (int c = 0; c < 3; c++)
                {
                    // Every Channel       :D shm[out_col+filterCol][out_row+filterRow]
                    sum += filter_c[filterRow * filter_dim + filterCol] * sh_mem[(threadIdx.y + filterRow) * blockDim.x + (threadIdx.x + filterCol) * IMAGE_CHANNELS + c];
                    // printf("%d\n",((threadIdx.y + filterRow) * blockDim.x + (threadIdx.x + filterCol)) * IMAGE_CHANNELS + c);
                    // if(((threadIdx.y + filterRow) * blockDim.x + (threadIdx.x + filterCol) * IMAGE_CHANNELS + c) >=972){
                    // printf("EORRR\n");
                    // }
                }
            }
        }

        output_image[out_row * blockDim.x + out_col] = sum;
    }
    else
    {
        // printf("%d\n", out_row);
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

    // 2. Copy Filter to Constant Memory
    cudaMemcpyToSymbol(filter_c, filter, filter_dim * filter_dim * sizeof(float));

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
                    read_image(input_dir, ent->d_name, &image_data, &width, &height, &channels);

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
                // (batch_counter + threads_per_block.z - 1) / threads_per_block.z);
                1);

            // printf("%d\n", blocks_num.x);
            // printf("%d\n", blocks_num.y);
            // printf("%d\n", input_tile_dim);
            // return 0;

            output_tile_convolution<<<blocks_num, threads_per_block, input_tile_dim * input_tile_dim * 3 * sizeof(float)>>>(d_batched_images, d_output, IMAGE_WIDTH, IMAGE_HEIGHT, batch_counter, filter_dim, input_tile_dim);

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
                write_image(output_dir, image_filenames[i], output + (i * IMAGE_HEIGHT * IMAGE_WIDTH), IMAGE_WIDTH, IMAGE_HEIGHT, 1);
            }

            // Free Host Memory
            free(output);

            if (batch_counter < batch_size)
            {
                break;
            }
        }

        // // Free Host Memory
        // // Free memory allocated for the filter in host memory
        // free(filter);
        // // Free memory allocated for the filter in host memory
        // free(image_filenames);

        // // Free Device Memory
        // // Free memory allocated for the filter in constant memory
        // cudaFree(filter_c);
        // // Free memory allocated for the batched images in device shared memory
        // cudaFree(d_batched_images);

        // Close the diresctory
        closedir(dir);
    }
    else
    {
        // Error opening directory
        perror("Unable to open directory");
    }
}