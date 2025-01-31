// # Vector addition in pure C (CPU-only execution)

# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <assert.h>

#define N 10000000
#define MAX_ERR 1e-6

 void vector_add(float *out, float *a, float *b, int n) {
     for(int i = 0; i < n; i++){
         out[i] = a[i] + b[i];
     }
 }

 int main(){
     printf("Hello\n");
     float *a, *b, *out;

     // Allocate memory
     a   = (float*)malloc(sizeof(float) * N);
     b   = (float*)malloc(sizeof(float) * N);
     out = (float*)malloc(sizeof(float) * N);

     // Initialize array
     for(int i = 0; i < N; i++){
         a[i] = 1.0f;
         b[i] = 2.0f;
     }

     // Main function
     vector_add(out, a, b, N);

     // Verification
     for(int i = 0; i < N; i++){
         assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
     printf("Computed Sucessfully");

    // To test print the result for addtion of first element
    printf("%f + %f\n",a[0],b[0]);
    printf("%f\n",out[0]);

    return 0;
}