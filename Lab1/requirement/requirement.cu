// nvcc -o requirement requirement.cu   
// requirement 3 2 1 2 3 4 5 6 

#include <stdio.h>
#include <stdlib.h>


char * concat_col( char** const col,const int nrows){
    //Function to Concatinate pointer to array of strings[char*] in one string
  
    // Computing new string length
    int new_str_len=0;
    for (int i=0; i<nrows; i++){
        new_str_len+=strlen(col[i]);
    }

    // Allocating Memory for the new String
    char *new_str = (char *)malloc(new_str_len * sizeof(char));
    
    // Filling new String
    int new_str_indx=0;
    for (int i=0;i<nrows;i++){
        // For each strinfg in the column
        int str_indx=0;
        while(col[i][str_indx] != '\0'){
            new_str[new_str_indx]=col[i][str_indx];
            new_str_indx++;
            str_indx++;
        }
    }

    // Adding Null terminator for the new_str
    new_str[new_str_indx]='\0';
    
    return new_str;
}

// Summation of String concatincation of the matrix cols
int main(int argc, char* argv[]){

    // Redaing Matrix Dimensions
    int nrows=atoi(argv[1]);
    int ncols=atoi(argv[2]);


    // Adjusting array to have pointer to the columns not the rows
    // Dynamic Array of Pointers
    char ***arr = (char ***)malloc(ncols * sizeof(char **));

    //Each elements is array of char*[string]
    for (int i = 0; i < ncols; i++) {
        arr[i] = (char **)malloc(nrows * sizeof(char*));
    }

    int args_conuter=3;

    // Reading Matrix Elements
    for(int i=0; i<nrows; i++){
        for(int j=0;j<ncols;j++){
            arr[j][i]=argv[args_conuter++];
        }
    }

    int result=0;
    for(int j=0;j<ncols;j++){
        // Get Conctaniantion of the col
        char* col_str= concat_col(arr[j],nrows);
        result+=atoi(col_str);
    }

    // Prining Result
    printf("%d",result);

    return 0;
}