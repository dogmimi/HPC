/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/

/******************************************************************************
* FIXED:
* 1: The segmentation fault is caused by the array size is too large to fit
*    into the stack. Actually if we put the program at different environment(OS),
*    the result would be different(some works fine, some crashed). What we can 
*    do is either modify our code structure or modify system's stack size.
*    For Linux, we can use 'ulimit -s x' to set stack size,(ex: x can be 100M)
*    Use 'ulimit -a' to see the default stack size to make sure it works!!
******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
  int nthreads, tid, i, j;
  double a[N][N];

  /* Fork a team of threads with explicit variable scoping */
  #pragma omp parallel shared(nthreads) private(i, j, tid, a)
  {

    /* Obtain/print thread info */
    tid = omp_get_thread_num();
    if(tid == 0) 
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
    printf("Thread %d starting...\n", tid);

    /* Each thread works on its own private copy of the array */
    for(i = 0; i < N; i++)
      for(j = 0; j < N; j++)
        a[i][j] = tid + i + j;

    /* For confirmation */
    printf("Thread %d done. Last element= %f\n", tid, a[N-1][N-1]);

  }  /* All threads join master thread and disband */
}

