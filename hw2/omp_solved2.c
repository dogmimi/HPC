/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/

/******************************************************************************
* FIXED:
* 1: this program doesn't specify each variable should be private or shared,
*    tid, total and i should be private, these variables should be unique
*    for each thread. 
* 2: add sumTotal variable to calculate total sum since total only contains
*    partial sum. Add sumTotal and critical to calculate the total sum.
* 3: modify type from float to double to increase precision.
******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
  int nthreads, i, tid;
  double total;
  double sumTotal = 0.0;

  /*** Spawn parallel region ***/
  #pragma omp parallel default(none) shared(nthreads, sumTotal) private(tid, total, i)
  {
    /* Obtain thread number */
    tid = omp_get_thread_num();
    /* Only master thread does this */
    if(tid == 0){
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
    printf("Thread %d is starting...\n", tid);
    
    #pragma omp barrier

    /* do some work */
    total = 0.0;
    #pragma omp for schedule(dynamic, 10)
    for(i = 0; i < 1000000; i++) 
      total = total + i * 1.0;
    
    #pragma omp critical
    {
      sumTotal += total;
    }
    printf ("Thread %d is done! Total= %e\n", tid, total);

  } /*** End of parallel region ***/
  
  printf ("Total= %e\n", sumTotal);
}
