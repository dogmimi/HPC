#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

/*  calculate determinant of matrix
 *  input: 
 *    mat: matrix with double elements
 *    n: dimensions of matrix mat 
 *  output:
 *    determinant of matrix mat 
*/

double determinant(double** mat, int n);

/* calcuate cofactor of matrix
 * input: 
 *   mat: matrix with double elements
 *   co: cofactor matrix(allocated by caller)
 *   n: dimensions of matrix mat
 * output:
 *   co: cofactor matrix
 * 
*/

void cofactor(double** mat, double** co, int n);

/* calculate transpose of matrix
 * input:
 *   mat:  matrix with double elements
 *   tran: transpose matrix(allocated by caller)
 *   n: dimensions of matrix mat
 * output:
 *   tran: transpose matrix
*/

void transpose(double** mat, double** tran, int n);

/* calcuate inverse of matrix
 * input:
 *   mat: matrix with double elements
 *   inv: inverse matrix(allocated by caller)
 *   n: dimensions of matrix mat
 * output:
 *   inv: inverse matrix
*/


void inverse(double** mat, double** inv, int n);

/* multiply n * n matrix by n * 1 matrix => n * 1 matrix
 * input:
 *   matA: n * n matrix with double elements
 *   matB: n * 1 matrix with double elements(only use 1D array to represent data)
 *   n :dimensions of matrix
 * output:
 *   matC: result, n * 1 matrix with double elements(only use 1D array to represent data)
*/

void multiplyMatrix(double** matA, double* matB, double* matC, int n);

/* minus n * 1 matrix A by n * 1 matrix B => n * 1 matrix A
 * input:
 *   matA: n * 1 matrix with double elements(only use 1D array to represent data)
 *   matB: n * 1 matrix with double elements(only use 1D array to represent data)
 *   matC: n * 1 matrix with double elements(only use 1D array to represent data)(allocated by caller)
 *   n :dimensions of matrix
 * output:
 *   matC: n * 1 matrix with double elements(only use 1D array to represent data)
*/

void minusMatrix(double* matA, double* matB, double* matC, int n);

/* norm of 1D array
 * input:
 *   arr: array
 *   n: array size n
 * output:
 *   norm of the vector
*/

double normOfArray(double* arr, int n);

/* check numbers' equality
 * input:
 *   a: number a
 *   b: number b
 * output:
 *   if a and b are almost equal
*/

bool almostEqual(double a, double b);

//----------------------IMPLEMENTATIONS----------------------

double determinant(double** mat, int n){
  if(n == 1){
    return mat[0][0];
  }else{
    double sign = 1.0;
    double det = 0.0;
    double** subMat = (double**)malloc(n * sizeof(double*));
    int i, j;
    for(i = 0; i < n; i++){
      subMat[i] = (double*)malloc(n * sizeof(double));
    }
    int c, p, q;
    int s = 1;
    for(c = 0; c < n; c++){
      p = 0;
      q = 0;
      for(i = 0; i < n; i++){
        for(j = 0; j < n; j++){
          subMat[i][j] = 0.0;
          if(i != 0 && j != c){
            subMat[p][q] = mat[i][j];
            if(q < (n - 2)){
              q++;
            }else{
              q = 0;
              p++;
            }
          }
        }
      }
      if(!almostEqual(0.0, mat[0][c])){
        det += (double)s * mat[0][c] * determinant(subMat, n - 1);
      }
      s *= -1;
    }

    for(i = 0; i < n; i++){
      free(subMat[i]);
    }
    free(subMat);
    return det; 
  }
}

void cofactor(double** mat, double** co, int n){
  int i, j, p, q, s, t;
  double det;
  double** subMat = (double**)malloc((n - 1) * sizeof(double*));
  for(i = 0; i < n - 1; i++){
    subMat[i] = (double*)malloc((n - 1) * sizeof(double));
  }

  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      s = 0;
      t = 0;

      for(p = 0; p < n; p++){
        for(q = 0; q < n; q++){
          if(i != p && j != q){
            subMat[s][t] = mat[p][q];
            if(t < (n - 2)){
              t++;
            }else{
              t = 0;
              s++;
            }
          }
        }
      } 
      det = determinant(subMat, n - 1);
      co[i][j] = pow(-1.0, i + j) * det;
    }
  }

  for(i = 0; i < n - 1; i++){
    free(subMat[i]);
  }
  free(subMat);
}

void transpose(double** mat, double** tran, int n){
  int i, j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      tran[i][j] = mat[j][i];
    }
  }
}

void inverse(double** mat, double** inv, int n){
  int i, j;
  double det = determinant(mat, n);
  double** tempMat = (double**)malloc(n * sizeof(double*));
  for(i = 0; i < n; i++){
    tempMat[i] = (double*)malloc(n * sizeof(double));
  }
  cofactor(mat, tempMat, n);
  transpose(tempMat, inv, n);
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      inv[i][j] /= det; 
    }
  } 
  for(i = 0; i < n; i++){
    free(tempMat[i]);
  }
  free(tempMat);
}

void multiplyMatrix(double** matA, double* matB, double* matC, int n){
  int i, j, k;
  double result;
  for(i = 0; i < n; i++){
    result = 0.0; 
    for(j = 0; j < n; j++){
      result += matA[j][i] * matB[j]; 
    }
    matC[i] = result;
  } 
}

void minusMatrix(double* matA, double* matB, double* matC, int n){
  int i;
  for(i = 0; i < n; i++){
    matC[i] = matA[i] - matB[i];
  }
}

double normOfArray(double* arr, int n){
  int i;
  double sum = 0.0;
  for(i = 0; i < n; i++){
    sum += (arr[i] * arr[i]);
  }
  return sqrt(sum);
}

bool almostEqual(double a, double b){
  const double epsilon = 0.000000000000001;
  return abs(a - b) <= epsilon * abs(a);
}

#endif
