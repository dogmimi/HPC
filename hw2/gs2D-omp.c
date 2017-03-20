/*
 * Author: HungWeiChen, hc2264
 * Description: This jacobi 2D smoothing is based on jacobi 1D iteration code provided
 *              by professor.
 * Parallel parts: 1. Main smoothing iteration(red points & black points)
 *                 2. calculate residual
 *                 3. allocate / release memory
*/

#include <stdio.h>
#include <math.h>
#include "util.h"
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double **u, int N, double invhsq)
{
  int i, j;
  double tmp, res = 0.0;

  #pragma omp parallel for default(none) shared(u, N, invhsq) private(i, j, tmp) reduction(+:res)
  for(i = 1; i <= N; i++){
    for(j = 1; j <= N; j++){
      tmp = ((4.0 * u[i][j] - u[i - 1][j] - u[i + 1][j] - u[i][j - 1] - u[i][j + 1]) * invhsq - 1.0);
      res += (tmp * tmp);
    }
  }

  return sqrt(res);
}


int main(int argc, char * argv[]){

  int i, j, N, iter, max_iters;

  max_iters = 1000;
  N = 1000;
  if(argc != 3){
    printf("Parameters count not matched!!! --> use default parameters\n");
    printf("should be ./executable N max_iterations\n");
  }else if(argc == 3){
    N = atoi(argv[1]);
    max_iters = atoi(argv[2]);
  }

  #pragma omp parallel
  {
    #ifdef _OPENMP
      int my_threadnum = omp_get_thread_num();
      int numthreads = omp_get_num_threads();
    #else
      int my_threadnum = 0;
      int numthreads = 1;
    #endif
      printf("I'm thread %d out of %d\n", my_threadnum, numthreads);
  }

  /* timing */
  timestamp_type time1, time2;
  get_timestamp(&time1);

  /* Allocation of vectors, including left and right ghost points */
  double **u    = (double **) calloc(sizeof(double*), N + 2);
  
  #pragma omp parallel for default(none) shared(N, u) private(i)
  for(i = 0; i < N + 2; i++){
    u[i] = (double *) calloc(sizeof(double), N + 2);
  }

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1.0 / hsq;
  double res, res0, tol = 1e-5;

  /* initial residual */
  res0 = compute_residual(u, N, invhsq);
  res = res0;

  for(iter = 0; (iter < max_iters) && (res / res0 > tol); iter++){

    /* GS step for all the inner points */

    //red points
    /*
    #pragma omp parallel for default(none) shared(N, u, hsq) private(i, j)
    for(i = 1; i <= N; i++){
      for(j = 1; j <= N; j++){
        if((i + j) % 2 == 0){
          u[i][j] = 0.25 * (u[i - 1][j] + u[i][j - 1] + u[i + 1][j] + u[i][j + 1] + hsq);
        }
      }
    }*/
 
    #pragma omp parallel for default(none) shared(N, u, hsq) private(i, j)
    for(i = 1; i <= N; i++){
      for(j = (i % 2) + 1; j <= N; j += 2){
        u[i][j] = 0.25 * (u[i - 1][j] + u[i][j - 1] + u[i + 1][j] + u[i][j + 1] + hsq);
      } 
    }

    //black points
    /*
    #pragma omp parallel for default(none) shared(N, u, hsq) private(i, j)
    for(i = 1; i <= N; i++){
      for(j = 1; j <= N; j++){
        if((i + j) % 2 == 1){
          u[i][j] = 0.25 * (u[i - 1][j] + u[i][j - 1] + u[i + 1][j] + u[i][j + 1] + hsq);
        }
      }
    }*/

    #pragma omp parallel for default(none) shared(N, u, hsq) private(i, j)
    for(i = 1; i <= N; i++){
      for(j = ((i + 1) % 2) + 1; j <= N; j += 2){
        u[i][j] = 0.25 * (u[i - 1][j] + u[i][j - 1] + u[i + 1][j] + u[i][j + 1] + hsq);
      }
    }

    if(0 == (iter % 10)){
      res = compute_residual(u, N, invhsq);
      printf("Iter %d: Residual: %g\n", iter, res);
    }
  }

  /* Clean up */
  #pragma omp parallel for default(none) shared(N, u) private(i)
  for(i = 0; i < N + 2; i++){
    free(u[i]);
  }
    
  free(u);

  /* timing */
  get_timestamp(&time2);
  double elapsed = timestamp_diff_in_seconds(time1, time2);
  printf("Time elapsed is %f seconds.\n", elapsed);
  return 0;
}
