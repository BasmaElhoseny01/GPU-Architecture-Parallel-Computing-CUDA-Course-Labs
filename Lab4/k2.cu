// Match Input Tile
// All Threads Participate in the Loading of the Tile but some threads do not participate in the computation of the output tile
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
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_CHANNELS 3

#define OUTPUT_TILE_DIM 16
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

// Device Kernels
__global__ void BatchConvolution(float *image, float *output_image, int width, int height, int batch_size, int filter_dim)
{
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    int outDepth = blockDim.z * blockIdx.z + threadIdx.z;

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
                    sum += filter_c[filterRow * filter_dim + filterCol] * (float)image[(outDepth * height * width + inRow * width + inCol) * IMAGE_CHANNELS + c];
                }
                // }
            }
        }
        output_image[(outDepth * height * width) + (outRow * width) + outCol] = sum;
    }
}

__global__ void input_tile_convolution_gpt(float *image, float *output_image, int width, int height, int batch_size, int filter_dim)
{
    // Shared Memory
    __shared__ float shared_image[16][16][4];

    // Load the Tile
    int inRow = blockDim.y * blockIdx.y + threadIdx.y;
    int inCol = blockDim.x * blockIdx.x + threadIdx.x;
    int inDepth = blockDim.z * blockIdx.z + threadIdx.z;

    // Load the Tile
    if (inRow < height && inCol < width)
    {
        for (int c = 0; c < 3; c++)
        {
            shared_image[threadIdx.y][threadIdx.x][threadIdx.z] = image[(inDepth * height * width + inRow * width + inCol) * IMAGE_CHANNELS + c];
        }
    }

    __syncthreads();

    // Convolution
    int outRow = inRow;
    int outCol = inCol;
    int outDepth = inDepth;

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
                    sum += filter_c[filterRow * filter_dim + filterCol] * shared_image[inRow % 16][inCol % 16][c];
                }
                // }
            }
        }
        output_image[(outDepth * height * width) + (outRow * width) + outCol] = sum;
    }
}

__global__ void input_tile_convolution(float *image, float *output_image, int width, int height, int batch_size, int filter_dim)
{
    // Input Image Indices
    int in_row = blockDim.y * blockIdx.y + threadIdx.y;
    int in_col = blockDim.x * blockIdx.x + threadIdx.x;

    
    // OutPut Image Indices
    // 6*0+0-1 =>-1
    int out_row = OUTPUT_TILE_DIM * blockIdx.y + threadIdx.y - (filter_dim / 2);
    int out_col = OUTPUT_TILE_DIM * blockIdx.x + threadIdx.x - (filter_dim / 2);


    // Store all elements needed to compute output in shared memory for the 3 channels
    extern __shared__ float sh_mem[];

    // if (in_row<height &&in_col<width){
    // if ((OUTPUT_TILE_DIM * blockIdx.y + threadIdx.y)<(height+filter_dim / 2) &&(OUTPUT_TILE_DIM * blockIdx.x + threadIdx.x)<(width){
    if (out_row>=0 && out_row<height && out_col>=0 && out_col<width){
        // out_row=-1 won't load although its input is 0 --> This is the case of the border threads :D Padd to be added
        //  This Row and Col are in the input image boundary
        // Load the Tile
        // sh_mem[threadIdx.y][threadIdx.x] = image[in_row][in_col];  --> For Every Channel

        for (int c = 0; c < 3; c++)
        {
            sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c] = image[((out_row+filter_dim / 2) * width + (out_col+filter_dim / 2)) * IMAGE_CHANNELS + c];
            // sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c] = image[((out_row) * width + (out_col)) * IMAGE_CHANNELS + c];
            // sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c] = image[(in_row * width + in_col) * IMAGE_CHANNELS + c];
        }
    }
    else{
        // Padding
        for (int c = 0; c < 3; c++)
        {
            // sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c] = 0.0;
            sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c] = 0.0;
        }
    }
    // Wait for all threads to finish loading the tile
    __syncthreads();

    // if (out_row>=0 && out_row<height && out_col>=0 && out_col<width){
    // output_image[out_row*width + out_col]=sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + 0];
    // }
    
    // The inner Thread of the Blocks are to be active and border threads are to be inactive :D
    // Case filter raduis is 1 we will get border col -1 and OUTPUT_TILE_DIM --> This to be idel threads
    // - - - - - - - -
    // - X X X X X X -        X X X X X X 
    // - X X X X X X -        X X X X X X
    // - X X X X X X -        X X X X X X
    // - X X X X X X -  -->   X X X X X X
    // - X X X X X X -        X X X X X X
    // - X X X X X X -        X X X X X X
    // - - - - - - - -
    // These indices are Per Tile :D [NOT FOR THE WHOLE IMAGE]
    int out_tile_col = threadIdx.x - (filter_dim / 2);  // .. -1  ...  OUTPUT_TILE_DIM ..
    int out_tile_row = threadIdx.y - (filter_dim / 2);  // .. -1  ...  OUTPUT_TILE_DIM ..


    if (out_row>=0 && out_row<height && out_col>=0 && out_col<width){
        if (out_tile_col>=0 && out_tile_col<OUTPUT_TILE_DIM && out_tile_row>=0 && out_tile_row<OUTPUT_TILE_DIM)
        {

            float sum = 0;
            //  Now This Thread is Active and will compute the output for [out_row][out_col] :D
            // So it will take its input from the shared memory :D out_row - filter_raduis --> out_row + filter_raduis
            //                                                     out_col - filter_raduis --> out_col + filter_raduis

            // Looping over mask :D
            for (int filterRow = 0; filterRow < filter_dim; filterRow++)
            {
                for (int filterCol = 0; filterCol < filter_dim; filterCol++)
                {
                    // For Every Channel
                    for (int c = 0; c < 3; c++)
                    {
                        // Every Channel
                        // sum += filter_c[filterRow * filter_dim + filterCol] * sh_mem[out_tile_col-filter_raduis][out_tile_row-filter_raduis]
                        sum += filter_c[filterRow * filter_dim + filterCol] * sh_mem[((out_tile_row + filterRow) * blockDim.x + (out_tile_col + filterCol))*IMAGE_CHANNELS+c];
                    }
                }
            }        
            // output_image[out_row][out_col] = ..;
            output_image[out_row*width + out_col] = sum;

        }
        // Else Border Threads in the block (=input title) so idle :D
        if (out_row*width + out_col==0){
            printf("Thread Id: %d %d %d\n",threadIdx.x,threadIdx.y,threadIdx.z);
            printf( "Block Id: %d %d %d\n",blockIdx.x,blockIdx.y,blockIdx.z);
            printf("Out Row: %d Out Col: %d\n",out_row,out_col);
            printf("In Row: %d In Col: %d\n",in_row,in_col);

            printf("sum %f\n",output_image[out_row*width + out_col]);
            // output_image[out_row*width + out_col]=0.0;
        }
            if (in_row*width + in_col==0){
            printf("Thread Id: %d %d %d\n",threadIdx.x,threadIdx.y,threadIdx.z);
            printf( "Block Id: %d %d %d\n",blockIdx.x,blockIdx.y,blockIdx.z);
            printf("**Out Row: %d Out Col: %d\n",out_row,out_col);
            printf("In Row: %d In Col: %d\n",in_row,in_col);

            printf("sum %f\n",output_image[in_row*width + in_col]);
        }
    }
}

// __global__ void input_tile_convolution(float *image, float *output_image, int width, int height, int batch_size, int filter_dim)
// {
    
//     // Input Tile
//     // int in_row = blockDim.y * blockIdx.y + threadIdx.y - (filter_dim / 2);
//     // int in_col = blockDim.x * blockIdx.x + threadIdx.x - (filter_dim / 2);
//     int in_row = OUTPUT_TILE_DIM * blockIdx.y + threadIdx.y - (filter_dim / 2);
//     int in_col = OUTPUT_TILE_DIM * blockIdx.x + threadIdx.x - (filter_dim / 2);
//     int in_depth = blockDim.z * blockIdx.z + threadIdx.z;

//     // Output Tile
//     int out_row = blockIdx.y * OUTPUT_TILE_DIM + threadIdx.y;
//     int out_col = blockIdx.x * OUTPUT_TILE_DIM + threadIdx.x;

//     // Store all elements needed to compute output in shared memory for the 3 channels
//     extern __shared__ float sh_mem[];

//     // Load the Tile
//     if (in_row < height && in_col < width && in_row >= 0 && in_col >= 0)

//     {
//         for (int c = 0; c < 3; c++)
//         {
//             sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c] = image[(in_row * width + in_col) * IMAGE_CHANNELS + c];
//         }
//     }
//     else{
//         // Add Zeros For Padding :D
//         for (int c = 0; c < 3; c++)
//         {
//             sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS + c] = 0;
//         }
//     }

//     // else
//     // {
//     //     sh_mem[(threadIdx.y * blockDim.x + threadIdx.x) * IMAGE_CHANNELS] = 0;
//     // }
//     __syncthreads();
//     /////////////////////////////////////////////////

//     // if (in_row < height && in_col < width && in_row >= 0 && in_col >= 0)
//     if (threadIdx.x < OUTPUT_TILE_DIM && threadIdx.y < OUTPUT_TILE_DIM && out_row < height && out_col < width)
//     {

//         float sum = 0;

//         for (int filterRow = 0; filterRow < filter_dim; filterRow++)
//         {
//             for (int filterCol = 0; filterCol < filter_dim; filterCol++)
//             {
//                 // apply boundary conditions (ghost cells)
//                 int sh_row = max(0, min(threadIdx.y + filterRow, height - 1));
//                 int sh_col = max(0, min(threadIdx.x + filterCol, width - 1));

//                 // int sh_row = threadIdx.y + filterRow;
//                 // int sh_col = threadIdx.x + filterCol;
//                 // Check if out of Bounday --> This is useless in case of padding

//                 for (int c = 0; c < 3; c++)
//                 {
//                     sum += filter_c[filterRow * filter_dim + filterCol] *
//                            sh_mem[((sh_row)*blockDim.x +
//                                    (sh_col)) *
//                                       IMAGE_CHANNELS +
//                                   c];
//                 }
//             }
//         }
//         output_image[out_row * width + out_col] = sum;
//         // }
//     }
// }

__host__ void verify_convolution()
{
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

            dim3 threads_per_block(input_tile_dim, input_tile_dim, 1);

            dim3 blocks_num(
                (IMAGE_WIDTH + OUTPUT_TILE_DIM - 1) / OUTPUT_TILE_DIM,
                (IMAGE_HEIGHT + OUTPUT_TILE_DIM - 1) / OUTPUT_TILE_DIM,
                // (batch_counter + threads_per_block.z - 1) / threads_per_block.z);
                1);

            input_tile_convolution<<<blocks_num, threads_per_block, input_tile_dim * input_tile_dim * 3 * sizeof(float)>>>(d_batched_images, d_output, IMAGE_WIDTH, IMAGE_HEIGHT, batch_counter, filter_dim);

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