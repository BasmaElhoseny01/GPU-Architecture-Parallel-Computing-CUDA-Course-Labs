#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#define MAX_ERR 1e-3

__global__ void vector_add(float *out, float *matrix, float *vector, int N, int M)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        for (int i = 0; i < M; i++)
        {
            out[tid] += vector[i] * matrix[(tid * M) + i];
        }
    }
}

int main(int argc, char *argv[])
{
    char *input_path = argv[1];
    char *output_path = argv[2];

    FILE *file = fopen(input_path, "r");

    if (!file)
    {
        printf("\n Unable to open : %s ", input_path);
        return -1;
    }

    char line[5000];

    int test_cases = atoi(fgets(line, sizeof(line), file));
    printf("test_cases: %d\n", test_cases);

    for (int k = 0; k < test_cases; k++)
    {
        int matrix_rows, matrix_columns;
        const char delimiter[] = " ";
        char *matrix_line = fgets(line, sizeof(line), file);

        matrix_rows = atoi(strtok(matrix_line, delimiter));
        matrix_columns = atoi(strtok(NULL, delimiter));
        printf("\nmatrix_rows: %d\n", matrix_rows);
        printf("matrix_columns: %d\n", matrix_columns);

        // Allocate host memory
        float *matrix = (float *)malloc(sizeof(float) * matrix_rows * matrix_columns);
        float *vector = (float *)malloc(sizeof(float) * matrix_columns);
        float *out = (float *)malloc(sizeof(float) * matrix_rows);

        // initialize out with zeros
        for (int i = 0; i < matrix_rows; i++)
        {
            out[i] = 0;
        }
        for (int i = 0; i < matrix_rows; i++)
        {
            char *matrix_line = fgets(line, sizeof(line), file);
            char *token = strtok(matrix_line, delimiter);
            for (int j = 0; j < matrix_columns; j++)
            {
                printf("%d", i * matrix_columns + j);
                matrix[(i * matrix_columns) + j] = atof(token);
                token = strtok(NULL, delimiter);
            }
        }

        // each vector element is in a new line
        for (int i = 0; i < matrix_columns; i++)
        {
            char *vector_line = fgets(line, sizeof(line), file);
            vector[i] = atof(vector_line);
        }

        printf("\nMatrix\n");
        // print the matrix and the vector
        for (int i = 0; i < matrix_rows * matrix_columns; i++)
        {
            printf("%f ", matrix[i]);
        }

        printf("\nVector\n");
        for (int i = 0; i < matrix_columns; i++)
        {
            printf("%f ", vector[i]);
        }
        printf("\n");

        float *d_matrix, *d_vector, *d_out;

        // Allocate device memory
        cudaMalloc((void **)&d_matrix, sizeof(float) * matrix_rows * matrix_columns);
        cudaMalloc((void **)&d_vector, sizeof(float) * matrix_columns);
        cudaMalloc((void **)&d_out, sizeof(float) * matrix_rows);

        // Transfer data from host to device memory
        cudaMemcpy(d_matrix, matrix, sizeof(float) * matrix_rows * matrix_columns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector, vector, sizeof(float) * matrix_columns, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, out, sizeof(float) * matrix_rows, cudaMemcpyHostToDevice);

        // Executing kernel
        vector_add<<<(matrix_rows + 255) / 256, 256>>>(d_out, d_matrix, d_vector, matrix_rows, matrix_columns);

        // Transfer data back to host memory
        cudaMemcpy(out, d_out, sizeof(float) * matrix_rows, cudaMemcpyDeviceToHost);

        for (int i = 0; i < matrix_rows; i++)
        {
            printf("\nout[%d] = %f", i, out[i]);
        }

        // Verification :D
        for (int i = 0; i < matrix_rows; i++)
        {
            // out[i]  result
            // lOOP on rows for the output vector
            float res = 0;
            for (int j = 0; j < matrix_columns; j++)
            {
                res += matrix[j + i * matrix_columns] * vector[j];
            }
            if (fabs(out[i] - res) >= MAX_ERR)
            {
                printf("Assertion failed: Maximum error exceeded!\n");
                printf("Computed value: %f\n", out[i]);
                printf("Reference value: %f\n", res);
                printf("Absolute error: %f\n", fabs(out[i] - res));
                printf("Check The Max Error: %f\n", MAX_ERR);
            }
            assert(fabs(out[i] - res) < MAX_ERR);
        }

        printf("\nPASSED\n");

        // Write Result to the file
        // print output to file, create if it doesn't exist and override it if it does
        FILE *output_file = fopen(output_path, "w");
        if (!output_file)
        {
            printf("\n Unable to open : %s ", output_path);
            return -1;
        }
        for (int i = 0; i < matrix_rows; i++)
        {
            fprintf(output_file, "%f\n", out[i]);
        }
        // fprintf(output_file, "\n");
        fclose(output_file);

        // Deallocate device memory
        cudaFree(d_matrix);
        cudaFree(d_vector);
        cudaFree(d_out);

        // Deallocate host memory
        free(matrix);
        free(vector);
        free(out);
    }
    fclose(file);
    return 0;
}