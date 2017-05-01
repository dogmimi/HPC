/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <cmath>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaProfiler.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>

#include <nvrtc_helper.h>
#include "../utility/util.h"

extern "C" void
launch_dummy();

extern "C" void
launch_rof(float* data, float* result, int width, int height);

extern "C" void
launch_tvl1(float* data, float* result, int width, int height);

/**
 * Host main routine
 */
int main(int argc, char **argv)
{
  //read image data from file
  int i, j, v, width, height, size, d_size;
  float* data;
  float* result; 
  FILE *inputFile;
  inputFile = fopen("../originalFile.txt", "r");
  fscanf(inputFile, "%d", &height);
  fscanf(inputFile, "%d", &width);
  printf("%d\n", height);
  printf("%d\n", width);
  size = width * height;
  d_size = sizeof(float) * size;
  
  data = (float *) calloc(sizeof(float), size);
  result = (float *) calloc(sizeof(float), size);
  for(i = 0; i < size; i++){
    fscanf(inputFile, "%d", &v);
    data[i] = (float)v / 255.0;
  }
  fclose(inputFile);

  //dummy execution to launch kernel first
  launch_dummy();

  //gpu execution
  timestamp_type tsStart, tsFinish;
  double elapsed;
  get_timestamp(&tsStart);
  launch_rof(data, result, width, height);
  get_timestamp(&tsFinish);
  elapsed = timestamp_diff_in_seconds(tsStart, tsFinish);
  printf("Time elapsed for CUDA ROF solver is %f seconds.\n", elapsed);

  //write image to file
  FILE *outputFile = fopen("../rofCudaResultFile.txt", "w+");
  fprintf(outputFile, "%d\n", height);
  fprintf(outputFile, "%d\n", width);
  for(i = 0; i < size; i++){
    fprintf(outputFile, "%f\n", result[i]);
  }
  fclose(outputFile);

  //gpu execution
  get_timestamp(&tsStart);
  launch_tvl1(data, result, width, height);
  get_timestamp(&tsFinish);
  elapsed = timestamp_diff_in_seconds(tsStart, tsFinish);
  printf("Time elapsed for CUDA TVL1 solver is %f seconds.\n", elapsed);

  //write image to file
  outputFile = fopen("../tvl1CudaResultFile.txt", "w+");
  fprintf(outputFile, "%d\n", height);
  fprintf(outputFile, "%d\n", width);
  for(i = 0; i < size; i++){
    fprintf(outputFile, "%f\n", result[i]);
  }
  fclose(outputFile);

  //release resources

  free(data);
  free(result);

  return 0;
}
