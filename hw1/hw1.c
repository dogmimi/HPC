#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "helper.h"
#include "util.h"

#define FACTOR 10000

void solveByJacobi(int n, int iterations, int factor){
  //initialization
  int i, j;
  int count = 0;
  double norm, firstNorm;
  /* 
  double** A = (double**)calloc(n, sizeof(double*));
  for(i = 0; i < n; i++){
    A[i] = (double*)calloc(n, sizeof(double));
  }
  for(i = 0; i < n; i++){
    A[i][i] = 2.0;
  }
  for(i = 0; i < (n - 1); i++){
    A[i + 1][i] = -1.0;
    A[i][i + 1] = -1.0;
  }*/
  double* u = (double*)calloc(n, sizeof(double));
  for(i = 0; i < n; i++){
    u[i] = 0.0;
  }
  double* v = (double*)calloc(n, sizeof(double)); //temp array to do iterations
  double* f = (double*)calloc(n, sizeof(double));
  double* temp = (double*)calloc(n, sizeof(double)); //temp array to do norm 
  
  for(i = 0; i < n; i++){
    f[i] = 1.0 / (double)((n + 1) * (n + 1));
  }

  //solve
  double* iterResults = u;
  double* nextIterResults = v;

  for(i = 0; i < n; i++){
    nextIterResults[i] = f[i];
    double sum = 0.0;
    /*
    for(j = 0; j < i; j++){
      sum += A[i][j] * iterResults[j]; 
    }
    for(j = i + 1; j < n; j++){
      sum += A[i][j] * iterResults[j]; 
    }
    */
    if(i - 1 >= 0){
      sum -= iterResults[i - 1]; 
    }
    if(i + 1 < n){
      sum -= iterResults[i + 1]; 
    }

    nextIterResults[i] -= sum;
    //nextIterResults[i] /= A[i][i];
    nextIterResults[i] /= 2.0;
  }

  //calculate first norm to compare
  //multiplyMatrix(A, nextIterResults, iterResults, n);
  iterResults[0] = 2.0 * nextIterResults[0] - nextIterResults[1];
  for(j = 1; j < n - 1; j++){
    iterResults[j] = 2.0 * nextIterResults[j] - nextIterResults[j - 1] - nextIterResults[j + 1];
  }
  iterResults[n - 1] = 2.0 * nextIterResults[n - 1] - nextIterResults[n - 2];
  minusMatrix(iterResults, f, temp, n);
  firstNorm = normOfArray(temp, n);
  norm = firstNorm;
  printf("first norm: %.12lf\n", norm);

  while((count < iterations) && ((firstNorm / norm) < factor)){
    if(count % 2 == 0){
      iterResults = v;
      nextIterResults = u;
    }else{
      iterResults = u;
      nextIterResults = v;
    } 

    //duplicated code, need to refactor later!
    for(i = 0; i < n; i++){
      nextIterResults[i] = f[i];
      double sum = 0.0;
      /*
      for(j = 0; j < i; j++){
        sum += A[i][j] * iterResults[j]; 
      }
      for(j = i + 1; j < n; j++){
        sum += A[i][j] * iterResults[j]; 
      }
      */
      if(i - 1 >= 0){
        sum -= iterResults[i - 1]; 
      }
      if(i + 1 < n){
        sum -= iterResults[i + 1]; 
      }

      nextIterResults[i] -= sum;
      //nextIterResults[i] /= A[i][i];
      nextIterResults[i] /= 2.0;
    }
    //calculate first norm to compare
    //print1DMatrix(nextIterResults, n);
    //multiplyMatrix(A, nextIterResults, iterResults, n);
    iterResults[0] = 2.0 * nextIterResults[0] - nextIterResults[1];
    for(j = 1; j < n - 1; j++){
      iterResults[j] = 2.0 * nextIterResults[j] - nextIterResults[j - 1] - nextIterResults[j + 1];
    }
    iterResults[n - 1] = 2.0 * nextIterResults[n - 1] - nextIterResults[n - 2];

    minusMatrix(iterResults, f, temp, n);
    norm = normOfArray(temp, n);
    //printf("#%d iteration norm: .%12lf\n", count, norm);

    count++;
  }
  printf("#%d iteration norm: %.12lf\n", count, norm);
   

  //release resources
  //A
  /*for(i = 0; i < n; i++){
    free(A[i]);
  }
  free(A);*/
  //u
  free(u);
  //v
  free(v);
  //f
  free(f);
  //temp
  free(temp);
}

void solveByGauss(int n, int iterations, int factor){
  //initialization
  int i, j;
  int count = 0;
  double norm, firstNorm;
 
  /* 
  double** A = (double**)calloc(n, sizeof(double*));
  for(i = 0; i < n; i++){
    A[i] = (double*)calloc(n, sizeof(double));
  }
  for(i = 0; i < n; i++){
    A[i][i] = 2.0;
  }
  for(i = 0; i < (n - 1); i++){
    A[i + 1][i] = -1.0;
    A[i][i + 1] = -1.0;
  }*/
  double* u = (double*)calloc(n, sizeof(double));
  for(i = 0; i < n; i++){
    u[i] = 0.0;
  }
  double* v = (double*)calloc(n, sizeof(double)); //temp array to do iterations
  double* f = (double*)calloc(n, sizeof(double));
  double* temp = (double*)calloc(n, sizeof(double)); //temp array to do norm 
  
  for(i = 0; i < n; i++){
    f[i] = 1.0 / (double)((n + 1) * (n + 1));
  }

  //solve
  double* iterResults = u;
  double* nextIterResults = v;

  for(i = 0; i < n; i++){
    nextIterResults[i] = f[i];
    double sum = 0.0;
    /*
    for(j = 0; j < i; j++){
      sum += A[i][j] * nextIterResults[j]; 
    }
    for(j = i + 1; j < n; j++){
      sum += A[i][j] * iterResults[j]; 
    }*/
    if(i - 1 >= 0){
      sum -= nextIterResults[i - 1]; 
    }
    if(i + 1 < n){
      sum -= iterResults[i + 1]; 
    }

    nextIterResults[i] -= sum;
    //nextIterResults[i] /= A[i][i];
    nextIterResults[i] /= 2.0;
  }

  //calculate first norm to compare
  //multiplyMatrix(A, nextIterResults, iterResults, n);
  iterResults[0] = 2.0 * nextIterResults[0] - nextIterResults[1];
  for(j = 1; j < n - 1; j++){
    iterResults[j] = 2.0 * nextIterResults[j] - nextIterResults[j - 1] - nextIterResults[j + 1];
  }
  iterResults[n - 1] = 2.0 * nextIterResults[n - 1] - nextIterResults[n - 2];

  minusMatrix(iterResults, f, temp, n);
  firstNorm = normOfArray(temp, n);
  norm = firstNorm;
  printf("first norm: %.12lf\n", norm);

  while((count < iterations) && ((firstNorm / norm) < factor)){
    if(count % 2 == 0){
      iterResults = v;
      nextIterResults = u;
    }else{
      iterResults = u;
      nextIterResults = v;
    } 

    //duplicated code, need to refactor later!
    for(i = 0; i < n; i++){
      nextIterResults[i] = f[i];
      double sum = 0.0;
      /*
      for(j = 0; j < i; j++){
        sum += A[i][j] * nextIterResults[j]; 
      }
      for(j = i + 1; j < n; j++){
        sum += A[i][j] * iterResults[j]; 
      }*/
      if(i - 1 >= 0){
        sum -= nextIterResults[i - 1]; 
      }
      if(i + 1 < n){
        sum -= iterResults[i + 1]; 
      }
      nextIterResults[i] -= sum;
      //nextIterResults[i] /= A[i][i];
      nextIterResults[i] /= 2.0;
    }
    //calculate first norm to compare
    //print1DMatrix(nextIterResults, n);
    //multiplyMatrix(A, nextIterResults, iterResults, n);
    iterResults[0] = 2.0 * nextIterResults[0] - nextIterResults[1];
    for(j = 1; j < n - 1; j++){
      iterResults[j] = 2.0 * nextIterResults[j] - nextIterResults[j - 1] - nextIterResults[j + 1];
    }
    iterResults[n - 1] = 2.0 * nextIterResults[n - 1] - nextIterResults[n - 2];
    minusMatrix(iterResults, f, temp, n);
    norm = normOfArray(temp, n);
    //printf("#%d iteration norm: %.12lf\n", count, norm);

    count++;
  }
  printf("#%d iteration norm: %.12lf\n", count, norm);
   

  //release resources
  //A
  /*
  for(i = 0; i < n; i++){
    free(A[i]);
  }
  free(A);*/
  //u
  free(u);
  //v
  free(v);
  //f
  free(f);
  //temp
  free(temp);
}

int main(int argc, char ** argv){
  printf("method 0: Jaboci, method 1: Gauss-Seidel\n");
  int method = 0;
  int n = 100;
  int iterations = 1000;
  if(argc != 4){
    printf("Not enough parameters!! --> use default parameters\n");
  }else{
    method = atoi(argv[1]);
    n = atoi(argv[2]); 
    iterations = atoi(argv[3]);
  }
  printf("method = %d, n = %d, iterations = %d\n", method, n, iterations);
  int i;
  int factor = FACTOR;
  timestamp_type timeStart, timeEnd;
  //method : 0 -> Jaboci
  //method : 1 -> Gauss-Seidel
  
  if(method == 0){
    get_timestamp(&timeStart); 
    solveByJacobi(n, iterations, factor);
    get_timestamp(&timeEnd); 
    double timeElapsed = timestamp_diff_in_seconds(timeStart, timeEnd);
    printf("Jacobi time elapsed is %f seconds.\n", timeElapsed);
  }else if(method == 1){
    get_timestamp(&timeStart); 
    solveByGauss(n, iterations, factor);
    get_timestamp(&timeEnd); 
    double timeElapsed = timestamp_diff_in_seconds(timeStart, timeEnd);
    printf("Gauss-Seidel time elapsed is %f seconds.\n", timeElapsed);

  }else{
    printf("Linear Method is NOT supported!\n");
  }

  return 0;
}
