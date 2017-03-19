/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/

/******************************************************************************
* FIXED:
* 1: First problem is that the program can't be compiled because sum can't be
*    private, it can only be shared for reduction. Put it to correct scoped
*    can fix this. And since dotprod does not need to return value, change
*    its return type from float to void.
******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];
float sum;

void dotprod()
{
  int i, tid;

  tid = omp_get_thread_num();
  #pragma omp for reduction(+:sum)
  for(i = 0; i < VECLEN; i++)
  {
    sum = sum + (a[i] * b[i]);
    printf("tid= %d i=%d\n", tid, i);
  }
}

int main (int argc, char *argv[]) {
  int i;

  for(i = 0; i < VECLEN; i++)
    a[i] = b[i] = 1.0 * i;
  sum = 0.0;

  #pragma omp parallel
    dotprod();

  printf("Sum = %f\n", sum);
}

