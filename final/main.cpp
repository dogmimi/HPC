#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
//#include <cuda_runtime_api.h>
//#include <cuda.h>
#include "core/solver.h"
#include "utility/imageutility.h"
#include "utility/util.h"
#include <algorithm>

using namespace cv;

int main(){
/*  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess){
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if(deviceCount == 0){
    printf("There are no available device(s) that support CUDA\n");
  }else{
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }
*/
  //1. create a 2D double array to represent gray picture
  int height;
  int width;
  int depth;
  int channels;
  //string fileName = "nena.jpg";
  string fileName = "lenaLarge.jpg";
  if(!getImageInfo(fileName, width, height, channels, depth)){
    printf("get image info failed!!\n");
  }

  double** fImagePtr = new double*[height];
  double** fResultImagePtr = new double*[height];
  for(int i = 0; i < height; i++){
    fImagePtr[i] = new double[width];
    fResultImagePtr[i] = new double[width];
  }
  if(!readImageFromFile(fImagePtr, fileName)){
    printf("read image failed!!\n");
  }

  timestamp_type tsStart, tsFinish;
  double elapsed;
  size_t pos = fileName.find(".");

  //2. use total variation method to denoise 
  get_timestamp(&tsStart);
  solve_rof(fImagePtr, width, height, 1, fResultImagePtr);
  get_timestamp(&tsFinish);
  elapsed = timestamp_diff_in_seconds(tsStart, tsFinish);
  printf("Time elapsed for ROF solver is %f seconds.\n", elapsed);
  saveImageToFile(fResultImagePtr, width, height, "rof_" + fileName.erase(pos, fileName.length() - pos) + ".png");

  get_timestamp(&tsStart);
  solve_tvl1(fImagePtr, width, height, 1, fResultImagePtr);
  get_timestamp(&tsFinish);
  elapsed = timestamp_diff_in_seconds(tsStart, tsFinish);
  printf("Time elapsed for TVL1 solver is %f seconds.\n", elapsed);
  saveImageToFile(fResultImagePtr, width, height, "tvl1_" + fileName.erase(pos, fileName.length() - pos) + ".png");
   
  get_timestamp(&tsStart);
  solve_rof_arranged(fImagePtr, width, height, 1, fResultImagePtr);
  get_timestamp(&tsFinish);
  elapsed = timestamp_diff_in_seconds(tsStart, tsFinish);
  printf("Time elapsed for ROF solver(arranged) is %f seconds.\n", elapsed);
  saveImageToFile(fResultImagePtr, width, height, "rof_arranged_" + fileName.erase(pos, fileName.length() - pos) + ".png");

  get_timestamp(&tsStart);
  solve_tvl1_arranged(fImagePtr, width, height, 1, fResultImagePtr);
  get_timestamp(&tsFinish);
  elapsed = timestamp_diff_in_seconds(tsStart, tsFinish);
  printf("Time elapsed for TVL1 solver(arranged) is %f seconds.\n", elapsed);
  saveImageToFile(fResultImagePtr, width, height, "tvl1_arranged_" + fileName.erase(pos, fileName.length() - pos) + ".png");


  //3. release resources
  delete [] fImagePtr;
  delete [] fResultImagePtr;

  //Miscellaneous
  double **data = (double **) calloc(sizeof(double*), height);
  for(int i = 0; i < height; i++){
    data[i] = (double *) calloc(sizeof(double), width);
  }
  ReadFileTo2DArray("rofSSEResultFile.txt", data);
  saveImageToFile(data, width, height, "rof_mpi_sse_" + fileName.erase(pos, fileName.length() - pos) + ".png");

  ReadFileTo2DArray("tvl1SSEResultFile.txt", data);
  saveImageToFile(data, width, height, "tvl1_mpi_sse_" + fileName.erase(pos, fileName.length() - pos) + ".png");

  ReadFileTo2DArray("rofAvxResultFile.txt", data);
  saveImageToFile(data, width, height, "rof_mpi_avx_" + fileName.erase(pos, fileName.length() - pos) + ".png");

  ReadFileTo2DArray("tvl1AvxResultFile.txt", data);
  saveImageToFile(data, width, height, "tvl1_mpi_avx_" + fileName.erase(pos, fileName.length() - pos) + ".png");

  ReadFileTo2DArray("rofCudaResultFile.txt", data);
  saveImageToFile(data, width, height, "rof_cuda_" + fileName.erase(pos, fileName.length() - pos) + ".png");

  ReadFileTo2DArray("tvl1CudaResultFile.txt", data);
  saveImageToFile(data, width, height, "tvl1_cuda_" + fileName.erase(pos, fileName.length() - pos) + ".png");

  for(int i = 0; i < height; i++){
    free(data[i]);
  }
  free(data); 
  
  return 0;
}
