// nvcc -o out_1  ./k1.cu
// out_1 ./input ./filter.txt
// nvprof out_2 ./testfile.txt ./out.txt


#include <stdio.h>
#include <stdlib.h>

#define DIRENT_IMPLEMENTATION
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define FILTER_DIM 3
#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256

__host__ void read_filter(const char* file_path){
    FILE* file = fopen(file_path, "r");
    if(file == NULL){
        printf("Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    int n;
    fscanf(file, "%d", &n);
    printf("Filter size: %d\n", n);

    printf("Filter Applied: %d\n", n);
    unsigned char* filter = (unsigned char*)malloc(n * n * sizeof(unsigned char));
    for(int i = 0; i < n * n; i++){
        fscanf(file, "%hhu", &filter[i]);
        printf("%hhu ", filter[i]);

        if (i % n == n-1){
            printf("\n");
        }
    }


    fclose(file);

}


__host__ unsigned char* read_image(const char* file_path){
    printf("Reading image at %s\n", file_path);

    int width, height, channels;
    unsigned char* image = stbi_load(file_path, &width, &height, &channels, 0);
    if(image == NULL){
        printf("Error: Unable to open file %s\n", file_path);
        exit(1);
    }

    printf("Image size: %d x %d\n", width, height);
    printf("Channels: %d\n", channels);

    // stbi_image_free(image);

    return image;
}
int main(int argc, char* argv[]){

    // Input Arguments 
    char* input_dir=argv[1];
    // char* output_folder_pth=argv[2];
    int batch_size=1;
    char* filter_pth=argv[2];

    // 1. Reading Filter
    read_filter(filter_pth);


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
                char* file_name = ent->d_name;

                // Read Image
                // Concatenate directory path and filename
                char file_path[256];
                snprintf(file_path, sizeof(file_path), "%s/%s", input_dir, ent->d_name);
                unsigned char* image=read_image(file_path);

                
                // Process the file here
                // Example: Load image using ent->d_name
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