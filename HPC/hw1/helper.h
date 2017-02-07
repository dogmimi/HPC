#ifndef HELPER_H
#define HELPER_H

#include <stdlib.h>

/* print all matrix's elements
 * input:
 *   mat: matrix
 *   n: dimensions of matrix mat 
*/

void printMatrix(double** mat, int n);

/* print 1D matrix's elements
 * input:
 *   mat: matrix(only use 1D array to represent data)
 *   n: dimensions of matrix mat 
*/

void print1DMatrix(double* mat, int n);

void printMatrix(double** mat, int n){
  int i, j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      printf("%.3f ", mat[j][i]);
    }
    printf("\n");
  }
  printf("\n");
}

void print1DMatrix(double* mat, int n){
  int i;
  for(i = 0; i < n; i++){
    printf("%.3f ", mat[i]);
  }
  printf("\n");
}

#endif
