/*
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
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include "../utility/util.h"

float* d_input;
float* d_output;
float* d_dx;
float* d_dy;

texture<float, cudaTextureType1D, cudaReadModeElementType> tex_OriginalImage;
texture<float, cudaTextureType1D, cudaReadModeElementType> tex_Image;
texture<float, cudaTextureType1D, cudaReadModeElementType> tex_dx;
texture<float, cudaTextureType1D, cudaReadModeElementType> tex_dy;

__global__ void
dummy(){
}

__global__ void
test(float *input, float *output, int w, int h){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= w || y >= h){
    return;
  }
 
  int index = y * w + x;
  float texR = tex1Dfetch(tex_Image, index);
  float texlR = tex1Dfetch(tex_Image, index - 1);
  float texrR = tex1Dfetch(tex_Image, index + 1);
  //output[index] = input[index];
  output[index] = (texR + texlR + texrR) / 3;
}

__global__ void
nabla(float *dx, float *dy, int w, int h){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= w || y >= h){
    return;
  }
 
  int index = y * w + x;

  float v_dx = 0.0;
  float v_dy = 0.0;
  if(x < w - 1){
    v_dx = tex1Dfetch(tex_Image, index + 1) - tex1Dfetch(tex_Image, index);  
  }
  if(y < h - 1){
    v_dy = tex1Dfetch(tex_Image, index + w) - tex1Dfetch(tex_Image, index);  
  }

  dx[index] = v_dx;
  dy[index] = v_dy;
}

__global__ void
rof_part1(float *dx, float *dy, int w, int h, float L2, float Tau, float Sigma, float Theta){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= w || y >= h){
    return;
  }
 
  int index = y * w + x;

  float imagecopy = tex1Dfetch(tex_Image, index);
  float dx_v = tex1Dfetch(tex_dx, index);
  float dy_v = tex1Dfetch(tex_dy, index);
  float dx_inner_v = 0.0;
  float dy_inner_v = 0.0;
  float norm;

  if(x < w - 1){
    dx_inner_v = (tex1Dfetch(tex_Image, index + 1) - imagecopy) * Sigma;
  }
  if(y < h - 1){
    dy_inner_v = (tex1Dfetch(tex_Image, index + w) - imagecopy) * Sigma;
  }  
  dx_v += dx_inner_v; 
  dy_v += dy_inner_v;
  norm = max(sqrt(dx_v * dx_v + dy_v * dy_v), 1.0);
  dx_v /= norm; 
  dy_v /= norm;

  dx[index] = dx_v;
  dy[index] = dy_v;
}

__global__ void
rof_part2(float *output, int w, int h, float L2, float Tau, float Sigma, float Theta, float lt){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= w || y >= h){
    return;
  }
 
  int index = y * w + x;
  float nt = 0.0;
  float imagecopy = tex1Dfetch(tex_Image, index);
  if(x <= w - 1){
    nt -= tex1Dfetch(tex_dx, index);
  }
  if(x >= 1){
    nt += tex1Dfetch(tex_dx, index - 1);
  }
  if(y < h - 1){
    nt -= tex1Dfetch(tex_dy, index);
  }
  if(y >= 1){
    nt += tex1Dfetch(tex_dy, index - w);
  }

  float x1 = (imagecopy - nt * Tau + lt * tex1Dfetch(tex_OriginalImage, index)) / (1.0 + lt);
  output[index] = x1 + Theta * (x1 - imagecopy);
}

__global__ void
tvl1_part2(float *output, int w, int h, float L2, float Tau, float Sigma, float Theta, float shrink){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= w || y >= h){
    return;
  }
 
  int index = y * w + x;
  float nt = 0.0;
  float imagecopy = tex1Dfetch(tex_Image, index);
  if(x < w - 1){
    nt -= tex1Dfetch(tex_dx, index);
  }
  if(x >= 1){
    nt += tex1Dfetch(tex_dx, index - 1);
  }
  if(y < h - 1){
    nt -= tex1Dfetch(tex_dy, index);
  }
  if(y >= 1){
    nt += tex1Dfetch(tex_dy, index - w);
  }

  float temp = imagecopy - nt * Tau;
  float x1 = temp + max(min(tex1Dfetch(tex_OriginalImage, index) - temp, shrink), -shrink);
  output[index] = x1 + Theta * (x1 - imagecopy);
}

extern "C" void
launch_rof(float* data, float* result, int width, int height){
  //allocate resouces
  //Allocate the device input and output 
  checkCudaErrors(cudaMalloc(&d_input, sizeof(float) * width * height));  
  checkCudaErrors(cudaMalloc(&d_output, sizeof(float) * width * height));
  checkCudaErrors(cudaMalloc(&d_dx, sizeof(float) * width * height));
  checkCudaErrors(cudaMalloc(&d_dy, sizeof(float) * width * height));

  //copy memory from host to device
  checkCudaErrors(cudaMemcpy(d_input, data, sizeof(float) * width * height, cudaMemcpyHostToDevice));

  //bind texture
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  checkCudaErrors(cudaBindTexture(0, tex_OriginalImage, d_input, desc, width * height * sizeof(float)));
  checkCudaErrors(cudaBindTexture(0, tex_Image, d_input, desc, width * height * sizeof(float)));

  //initialization
  {
    int threadCounts = 16;
    dim3 cudaBlockSize((width + threadCounts - 1) / threadCounts, (height + threadCounts - 1) / threadCounts, 1);
    dim3 cudaGridSize(threadCounts, threadCounts, 1);
    nabla<<<cudaBlockSize, cudaGridSize>>>(d_dx, d_dy, width, height);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaBindTexture(0, tex_dx, d_dx, desc, width * height * sizeof(float)));
    checkCudaErrors(cudaBindTexture(0, tex_dy, d_dy, desc, width * height * sizeof(float)));
  }
  
  double fL2 = 8.0;
  double fTau = 0.02; 
  double fSigma = 1.0 / (fL2 * fTau);  
  double fTheta = 1.0;
  double lt = 8.0 * fTau;
  //execution
  
  for(int i = 0; i < 101; i++){
    int threadCounts = 16;
    dim3 cudaBlockSize((width + threadCounts - 1) / threadCounts, (height + threadCounts - 1) / threadCounts, 1);
    dim3 cudaGridSize(threadCounts, threadCounts, 1);
    //part1
    rof_part1<<<cudaBlockSize, cudaGridSize>>>(d_dx, d_dy, width, height, fL2, fTau, fSigma, fTheta);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaBindTexture(0, tex_dx, d_dx, desc, width * height * sizeof(float)));
    checkCudaErrors(cudaBindTexture(0, tex_dy, d_dy, desc, width * height * sizeof(float)));

    //part2
    rof_part2<<<cudaBlockSize, cudaGridSize>>>(d_output, width, height, fL2, fTau, fSigma, fTheta, lt);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaBindTexture(0, tex_Image, d_output, desc, width * height * sizeof(float)));
  }

  //copy memory from device to host
  checkCudaErrors(cudaMemcpy(result, d_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost));

  //release resources
  checkCudaErrors(cudaUnbindTexture(tex_OriginalImage));
  checkCudaErrors(cudaUnbindTexture(tex_Image));
  checkCudaErrors(cudaUnbindTexture(tex_dx));
  checkCudaErrors(cudaUnbindTexture(tex_dy));
  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_output));
  checkCudaErrors(cudaFree(d_dx));
  checkCudaErrors(cudaFree(d_dy));
}

extern "C" void
launch_tvl1(float* data, float* result, int width, int height){
  //allocate resouces
  //Allocate the device input and output 
  checkCudaErrors(cudaMalloc(&d_input, sizeof(float) * width * height));  
  checkCudaErrors(cudaMalloc(&d_output, sizeof(float) * width * height));
  checkCudaErrors(cudaMalloc(&d_dx, sizeof(float) * width * height));
  checkCudaErrors(cudaMalloc(&d_dy, sizeof(float) * width * height));

  //copy memory from host to device
  checkCudaErrors(cudaMemcpy(d_input, data, sizeof(float) * width * height, cudaMemcpyHostToDevice));

  //bind texture
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  checkCudaErrors(cudaBindTexture(0, tex_OriginalImage, d_input, desc, width * height * sizeof(float)));
  checkCudaErrors(cudaBindTexture(0, tex_Image, d_input, desc, width * height * sizeof(float)));

  //initialization
  {
    int threadCounts = 16;
    dim3 cudaBlockSize((width + threadCounts - 1) / threadCounts, (height + threadCounts - 1) / threadCounts, 1);
    dim3 cudaGridSize(threadCounts, threadCounts, 1);
    nabla<<<cudaBlockSize, cudaGridSize>>>(d_dx, d_dy, width, height);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaBindTexture(0, tex_dx, d_dx, desc, width * height * sizeof(float)));
    checkCudaErrors(cudaBindTexture(0, tex_dy, d_dy, desc, width * height * sizeof(float)));
  }

  double fL2 = 8.0;
  double fTau = 0.02; 
  double fSigma = 1.0 / (fL2 * fTau);  
  double fTheta = 1.0;
  double shrink = 1.0 * fTau;
  //execution
  
  for(int i = 0; i < 101; i++){
    int threadCounts = 16;
    dim3 cudaBlockSize((width + threadCounts - 1) / threadCounts, (height + threadCounts - 1) / threadCounts, 1);
    dim3 cudaGridSize(threadCounts, threadCounts, 1);
    //part1
    rof_part1<<<cudaBlockSize, cudaGridSize>>>(d_dx, d_dy, width, height, fL2, fTau, fSigma, fTheta);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaBindTexture(0, tex_dx, d_dx, desc, width * height * sizeof(float)));
    checkCudaErrors(cudaBindTexture(0, tex_dy, d_dy, desc, width * height * sizeof(float)));

    //part2
    tvl1_part2<<<cudaBlockSize, cudaGridSize>>>(d_output, width, height, fL2, fTau, fSigma, fTheta, shrink);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaBindTexture(0, tex_Image, d_output, desc, width * height * sizeof(float)));
  }

  //copy memory from device to host
  checkCudaErrors(cudaMemcpy(result, d_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost));

  //release resources
  checkCudaErrors(cudaUnbindTexture(tex_OriginalImage));
  checkCudaErrors(cudaUnbindTexture(tex_Image));
  checkCudaErrors(cudaUnbindTexture(tex_dx));
  checkCudaErrors(cudaUnbindTexture(tex_dy));
  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_output));
  checkCudaErrors(cudaFree(d_dx));
  checkCudaErrors(cudaFree(d_dy));
}

extern "C" void
launch_dummy(){
  int threadCounts = 16;
  dim3 cudaBlockSize(1, 1, 1);
  dim3 cudaGridSize(threadCounts, threadCounts, 1);
  dummy<<<cudaBlockSize, cudaGridSize>>>();
  checkCudaErrors(cudaDeviceSynchronize());
}
