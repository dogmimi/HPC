#include <stdio.h>
#include <math.h>
#include "util.h"
#include <string.h>
#include <mpi.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double **lu, int lN, double invhsq){
  int i, j;
  double tmp, gres = 0.0, lres = 0.0;

  for(i = 1; i <= lN; i++){
    for(j = 1; j <= lN; j++){
      tmp = ((4.0 * lu[i][j] - lu[i - 1][j] - lu[i + 1][j] - lu[i][j - 1] - lu[i][j + 1]) * invhsq - 1.0);
      lres += (tmp * tmp);
    }
  }

  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){

  int i, j, N, iter, max_iters, lN, mpirank, p;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  int half = sqrt(p);
  max_iters = 1000;
  N = 1000;
  if(argc != 3){
    printf("Parameters count not matched!!! --> use default parameters\n");
    printf("should be ./executable N max_iterations\n");
  }else if(argc == 3){
    N = atoi(argv[1]);
    max_iters = atoi(argv[2]);
  }

  /* compute number of unknowns handled by each process */
  lN = N / half;
  MPI_Status status;
  MPI_Request request_out_left[lN], request_in_left[lN];
  MPI_Request request_out_right[lN], request_in_right[lN];
  MPI_Request request_out_top, request_in_top;
  MPI_Request request_out_bottom, request_in_bottom;
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  timestamp_type time1, time2;
  get_timestamp(&time1);

  /* Allocation of vectors, including left and right ghost points */
  double **lu    = (double **) calloc(sizeof(double*), lN + 2);
  double **lunew = (double **) calloc(sizeof(double*), lN + 2);
  for(i = 0; i < lN + 2; i++){
    lu[i] = (double *) calloc(sizeof(double), lN + 2);
    lunew[i] = (double *) calloc(sizeof(double), lN + 2);
  }

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1.0 / hsq;
  double res, res0, tol = 1e-5;

  /* initial residual */
  res0 = compute_residual(lu, lN, invhsq);
  res = res0;

  for(iter = 0; (iter < max_iters) && (res / res0 > tol); iter++){
    
    /* Jacobi step for the boundary points */
    //top & bottom
    for(i = 1; i <= lN; i++){
      lunew[1][i] = 0.25 * (lu[0][i] + lu[2][i] + lu[1][i + 1] + lu[1][i - 1] + hsq);
      lunew[lN][i] = 0.25 * (lu[lN - 1][i] + lu[lN + 1][i] + lu[lN][i + 1] + lu[lN][i - 1] + hsq);
    }
    //left & right
    for(i = 1; i <= lN; i++){
      lunew[i][1] = 0.25 * (lu[i - 1][1] + lu[i + 1][1] + lu[i][0] + lu[i][2] + hsq);
      lunew[i][lN] = 0.25 * (lu[i - 1][lN] + lu[i + 1][lN] + lu[i][lN - 1] + lu[i][lN + 1] + hsq);
    }

    int half = sqrt(p);

    int row = mpirank / half;
    int col = mpirank % half;

    if (col < half - 1) {
      /* If not the last process, send/recv bdry values to the right */
      for(i = 1; i <= lN; i++){
        MPI_Isend(&(lunew[i][lN]), 1, MPI_DOUBLE, mpirank+1, 125+lN+i, MPI_COMM_WORLD, &request_out_right[i-1]);
        MPI_Irecv(&(lunew[i][lN+1]), 1, MPI_DOUBLE, mpirank+1, 125+i, MPI_COMM_WORLD, &request_in_right[i-1]);
      }
    }
    if (col > 0) {
      /* If not the first process, send/recv bdry values to the left */
      for(i = 1; i <= lN; i++){
        MPI_Isend(&(lunew[i][1]), 1, MPI_DOUBLE, mpirank-1, 125+i, MPI_COMM_WORLD, &request_out_left[i-1]);
        MPI_Irecv(&(lunew[i][0]), 1, MPI_DOUBLE, mpirank-1, 125+lN+i, MPI_COMM_WORLD, &request_in_left[i-1]);
      }
    }
    if (row < half - 1) {
      /* If not the first process, send/recv bdry values to the bottom */
      MPI_Isend(&(lunew[lN][1]), lN, MPI_DOUBLE, mpirank+half, 123, MPI_COMM_WORLD, &request_out_bottom);
      MPI_Irecv(&(lunew[lN+1][1]), lN, MPI_DOUBLE, mpirank+half, 124, MPI_COMM_WORLD, &request_in_bottom);
    }
    if (row > 0) {
      /* If not the first process, send/recv bdry values to the top */
      MPI_Isend(&(lunew[1][1]), lN, MPI_DOUBLE, mpirank-half, 124, MPI_COMM_WORLD, &request_out_top);
      MPI_Irecv(&(lunew[0][1]), lN, MPI_DOUBLE, mpirank-half, 123, MPI_COMM_WORLD, &request_in_top);
    }

    /* Jacobi step for all the inner points */
    for(i = 2; i < lN; i++){
      for(j = 2; j < lN; j++){
        lunew[i][j] = 0.25 * (lu[i - 1][j] + lu[i][j - 1] + lu[i + 1][j] + lu[i][j + 1] + hsq);
      }
    }
 
    /* check if Isend/Irecv are done */
    if(col < half - 1){
      for(i = 1; i <= lN; i++){
        MPI_Wait(&request_out_right[i - 1], &status);
        MPI_Wait(&request_in_right[i - 1], &status);
      }
    }
    if(col > 0){
      for(i = 1; i <= lN; i++){
        MPI_Wait(&request_out_left[i - 1], &status);
        MPI_Wait(&request_in_left[i - 1], &status);
      }
    }
    if(row < half - 1){
      MPI_Wait(&request_out_bottom, &status);
      MPI_Wait(&request_in_bottom, &status);
    }
    if(row > 0){
      MPI_Wait(&request_out_top, &status);
      MPI_Wait(&request_in_top, &status);
    }

    /* copy new_u onto u */
    double **utemp;
    utemp = lu;
    lu = lunew;
    lunew = utemp;
    if(0 == (iter % 10)){
      res = compute_residual(lu, lN, invhsq);
      if(mpirank == 0){
        printf("Iter %d: Residual: %g\n", iter, res);
      }
    }
  }

  /* Clean up */
  for(i = 0; i < lN + 2; i++){
    free(lu[i]);
    free(lunew[i]);
  }
    
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  get_timestamp(&time2);
  double elapsed = timestamp_diff_in_seconds(time1, time2);
  if(mpirank == 0){
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  
  MPI_Finalize();
  return 0;
}
