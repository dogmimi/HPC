#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
//#include <smmintrin.h>
#include <xmmintrin.h>
#include "../utility/util.h"

#define ITER 101

int main( int argc, char *argv[]){
  int rank, num_tasks, width, height, i, j, v, iter;
  float **data;
  float *data1D;
  int partialHeight;
  //mpi initialization
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
  timestamp_type tsStart, tsFinish;
  float elapsed;

  //read data from file
  if(rank == 0){
    FILE *inputFile;
    inputFile = fopen("../originalFile.txt", "r");
    fscanf(inputFile, "%d", &height);
    fscanf(inputFile, "%d", &width);
    printf("%d\n", height);
    printf("%d\n", width);
    data = (float **) valloc(sizeof(float*) * height);
    //data = (float **) calloc(sizeof(float*), height);
    for(i = 0; i < height; i++){
     data[i] = (float *) valloc(sizeof(float) * width);
     //data[i] = (float *) calloc(sizeof(float), width);
    }
    for(i = 0; i < height; i++){
      for(j = 0; j < width; j++){
        fscanf(inputFile, "%d", &v);
        data[i][j] = (float)v / 255.0;
      }
    }
    fclose(inputFile);
  
    get_timestamp(&tsStart);
    partialHeight = height / num_tasks;
    //1D array data
    data1D = (float *) valloc(sizeof(float) * width * height);
    //data1D = (float *) calloc(sizeof(float), width * height);
    //copy 2D data to 1D data
    int index = 0;
    for(i = 0; i < height; i++){
      for(j = 0; j < width; j++){
        data1D[index] = data[i][j];
        index++;
      }
    }
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&partialHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //1D array local data
  float *localData1D = (float *) valloc(sizeof(float) * width * (partialHeight + 1));
  //float *localData1D = (float *) calloc(sizeof(float), width * (partialHeight + 1));

  //scatter data to each tasks
  MPI_Scatter(data1D, partialHeight * width, MPI_FLOAT, localData1D, partialHeight * width, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if(rank > 0){
    MPI_Send(localData1D, width, MPI_FLOAT, rank - 1, 127, MPI_COMM_WORLD);
  }
  if(rank < num_tasks - 1){
    MPI_Recv(&localData1D[partialHeight * width], width, MPI_FLOAT, rank + 1, 127, MPI_COMM_WORLD, &status);
  }
     
  //do rof mpi version
  {
    float clambda = 1.0;
    float fL2 = 8.0;
    float fTau = 0.02; 
    float fSigma = 1.0 / (fL2 * fTau);
    float *fTempMask = (float*) valloc(sizeof(float) * 4);
    float fTheta = 1.0;
    int iteration = ITER;
    float norm;
    float dx_v, dy_v, dx_inner_v, dy_inner_v, nt, x1, imagecopy;
    float **partialData = (float **) valloc(sizeof(float*) * (partialHeight + 1));
    float **partialDx = (float **) valloc(sizeof(float*) * (partialHeight));
    float **partialDy = (float **) valloc(sizeof(float*) * (partialHeight + 1));
    float **partialImageCopy = (float **) valloc(sizeof(float*) * (partialHeight + 1));
    //float **partialData = (float **) calloc(sizeof(float*), partialHeight + 1);
    //float **partialDx = (float **) calloc(sizeof(float*), partialHeight);
    //float **partialDy = (float **) calloc(sizeof(float*), partialHeight + 1);
    //float **partialImageCopy = (float **) calloc(sizeof(float*), partialHeight + 1);
    for(i = 0; i < partialHeight + 1; i++){
      partialData[i] = (float *) valloc(sizeof(float) * width);
      partialDy[i] = (float *) valloc(sizeof(float) * width);
      partialImageCopy[i] = (float *) valloc(sizeof(float) * width);
      //partialData[i] = (float *) calloc(sizeof(float), width);
      //partialDy[i] = (float *) calloc(sizeof(float), width);
      //partialImageCopy[i] = (float *) calloc(sizeof(float), width);
    }
    for(i = 0; i < partialHeight; i++){
      partialDx[i] = (float *) valloc(sizeof(float) * width);
      //partialDx[i] = (float *) calloc(sizeof(float), width);
    }

    //rearrange 1D data to 2D data
    int counter = 0;
    for(i = 0; i < partialHeight + 1; i++){
      for(j = 0; j < width; j++){
        partialData[i][j] = localData1D[counter];
        counter++;
      }
    }

    //initialize partial dx 
    for(i = 0; i < partialHeight; i++){
      for(j = 0; j < width; j++){
        partialDx[i][j] = 0.0;
      }
    }
    //initialize partial dy
    for(i = 0; i < partialHeight + 1; i++){
      for(j = 0; j < width; j++){
        partialDy[i][j] = 0.0;
      }
    }

    for(i = 0; i < partialHeight + 1; i++){
      memcpy(partialImageCopy[i], partialData[i], width * sizeof(float));
    }
 
    //x part
    for(i = 0 ; i < partialHeight; i++){
      for(j = 0 ; j < width - 1; j++){
        partialDx[i][j] = partialImageCopy[i][j + 1] - partialImageCopy[i][j];
      }
    }
    //y part
    int boundary;
    if(rank < num_tasks - 1){
      boundary = partialHeight;
    }else{
      boundary = partialHeight - 1;
    }

    for(i = 1 ; i <= boundary; i++){
      for(j = 0 ; j < width; j++){
        partialDy[i][j] = partialImageCopy[i][j] - partialImageCopy[i - 1][j]; 
      }
    }
    //P = nabla(X)

    float lt = clambda * fTau;
    double shrink = clambda * fTau;
    //lt = clambda * tau
    __m128 xmm_sigma_mask = _mm_set_ps(fSigma, fSigma, fSigma, fSigma);
    __m128 xmm_zero_mask = _mm_set_ps(0.0, 0.0, 0.0, 0.0);
    __m128 xmm_ones_mask = _mm_set_ps(1.0, 1.0, 1.0, 1.0);
    __m128 xmm_tau_mask = _mm_set_ps(fTau, fTau, fTau, fTau);
    __m128 xmm_lt_mask = _mm_set_ps(lt, lt, lt, lt);
    __m128 xmm_lt1_mask = _mm_set_ps(lt + 1.0, lt + 1.0, lt + 1.0, lt + 1.0);
    __m128 xmm_theta_mask = _mm_set_ps(fTheta, fTheta, fTheta, fTheta);
    __m128 xmm_shrink1_mask = _mm_set_ps(shrink, shrink, shrink, shrink);
    __m128 xmm_shrink2_mask = _mm_set_ps(-shrink, -shrink, -shrink, -shrink);
    __m128 xmm_norm1;
    __m128 xmm_norm2;
    for(iter = 0; iter < iteration; iter++){
      //transfer imageCopy to its neighbors
      if(rank > 0){
        MPI_Send(partialImageCopy[0], width, MPI_FLOAT, rank - 1, 125, MPI_COMM_WORLD);
      }
      if(rank < num_tasks - 1){
        MPI_Recv(partialImageCopy[partialHeight], width, MPI_FLOAT, rank + 1, 125, MPI_COMM_WORLD, &status);
      } 
      //transfer partial dy to its neighbors
      if(rank > 0){
        MPI_Recv(partialDy[0], width, MPI_FLOAT, rank - 1, 126, MPI_COMM_WORLD, &status);
      }
      if(rank < num_tasks - 1){
        MPI_Send(partialDy[partialHeight], width, MPI_FLOAT, rank + 1, 126, MPI_COMM_WORLD);
      } 
   
      //synchronize here to make sure all buffers we need to calculate is set up. 
      MPI_Barrier(MPI_COMM_WORLD);
      
      for(i = 0; i < partialHeight; i++){
        float* dx_ptr = partialDx[i];
        float* dy_ptr = partialDy[i + 1];
        float* imagecopy_ptr = partialImageCopy[i];
        float* shift_x_imagecopy_ptr = &(partialImageCopy[i][1]);
        float* shift_y_imagecopy_ptr = &(partialImageCopy[i + 1][0]);
        __m128 xmm_dx, xmm_dy, xmm_dx_inner_v, xmm_dy_inner_v, xmm_shift_x_imagecopy, xmm_shift_y_imagecopy, xmm_imagecopy;
        for(j = 0; j < width / 4 - 1; j++){//j stands for block index, block size is 4
          xmm_shift_x_imagecopy = _mm_loadu_ps(shift_x_imagecopy_ptr);
          xmm_imagecopy = _mm_loadu_ps(imagecopy_ptr);
          xmm_dx_inner_v = _mm_sub_ps(xmm_shift_x_imagecopy, xmm_imagecopy);
          xmm_dx_inner_v = _mm_mul_ps(xmm_dx_inner_v, xmm_sigma_mask);

          if(rank == (num_tasks - 1) && (i == (partialHeight - 1))){
            xmm_dy_inner_v = xmm_zero_mask;
          }else{
            xmm_shift_y_imagecopy = _mm_loadu_ps(shift_y_imagecopy_ptr);
            xmm_dy_inner_v = _mm_sub_ps(xmm_shift_y_imagecopy, xmm_imagecopy);
            xmm_dy_inner_v = _mm_mul_ps(xmm_dy_inner_v, xmm_sigma_mask);
          }

          xmm_dx = _mm_loadu_ps(dx_ptr);
          xmm_dy = _mm_loadu_ps(dy_ptr);
          /*if(i == 0 && j == 0){
            printf("dx_v: %f\n", dx_ptr[0]);
            printf("dy_v: %f\n", dy_ptr[0]);
             float test[4];
            _mm_store_ps(&test, xmm_dx);
            printf("dx_v: %f\n", test[0]);
            _mm_store_ps(&test, xmm_dy);
            printf("dy_v: %f\n", test[0]);
          }*/
          xmm_dx_inner_v = _mm_add_ps(xmm_dx, xmm_dx_inner_v);
          xmm_dy_inner_v = _mm_add_ps(xmm_dy, xmm_dy_inner_v);
          
          xmm_norm1 = _mm_mul_ps(xmm_dx_inner_v, xmm_dx_inner_v);
          xmm_norm2 = _mm_mul_ps(xmm_dy_inner_v, xmm_dy_inner_v);
          xmm_norm1 = _mm_add_ps(xmm_norm1, xmm_norm2);
          xmm_norm1 = _mm_sqrt_ps(xmm_norm1);
          xmm_norm1 = _mm_max_ps(xmm_norm1, xmm_ones_mask);
          xmm_dx_inner_v = _mm_div_ps(xmm_dx_inner_v, xmm_norm1);
          xmm_dy_inner_v = _mm_div_ps(xmm_dy_inner_v, xmm_norm1);
        
          _mm_store_ps(&partialDx[i][j * 4], xmm_dx_inner_v);
          _mm_store_ps(&partialDy[i + 1][j * 4], xmm_dy_inner_v);

          shift_x_imagecopy_ptr += 4;//shift 4 pixels
          shift_y_imagecopy_ptr += 4;//shift 4 pixels
          imagecopy_ptr += 4;//shift 4 pixels
          dx_ptr += 4;//shift 4 pixels
          dy_ptr += 4;//shift 4 pixels
        }
        //printf("!!\n");
        xmm_imagecopy = _mm_loadu_ps(imagecopy_ptr);
        int k = 0;
        int lastBlock = (width / 4 - 1) * 4;
        for(j = lastBlock; j < width - 1; j++){//here j stands for width index, only deal 3 pixels!
          fTempMask[k] = (partialImageCopy[i][j + 1] - imagecopy_ptr[k]) * fSigma;
          k++; 
        }
        fTempMask[3] = 0.0;
        xmm_dx_inner_v = _mm_load_ps(fTempMask);
        if(rank == (num_tasks - 1) && (i == (partialHeight - 1))){
          xmm_dy_inner_v = xmm_zero_mask;
        }else{
          xmm_shift_y_imagecopy = _mm_loadu_ps(shift_y_imagecopy_ptr);
          xmm_dy_inner_v = _mm_sub_ps(xmm_shift_y_imagecopy, xmm_imagecopy);
          xmm_dy_inner_v = _mm_mul_ps(xmm_dy_inner_v, xmm_sigma_mask);
        }
        //printf("##\n");

        xmm_dx = _mm_load_ps(dx_ptr);
        xmm_dy = _mm_load_ps(dy_ptr);
        xmm_dx_inner_v = _mm_add_ps(xmm_dx, xmm_dx_inner_v);
        xmm_dy_inner_v = _mm_add_ps(xmm_dy, xmm_dy_inner_v);

        xmm_norm1 = _mm_mul_ps(xmm_dx_inner_v, xmm_dx_inner_v);
        xmm_norm2 = _mm_mul_ps(xmm_dy_inner_v, xmm_dy_inner_v);
        xmm_norm1 = _mm_add_ps(xmm_norm1, xmm_norm2);
        xmm_norm1 = _mm_sqrt_ps(xmm_norm1);
        xmm_norm1 = _mm_max_ps(xmm_norm1, xmm_ones_mask);
        xmm_dx_inner_v = _mm_div_ps(xmm_dx_inner_v, xmm_norm1);
        xmm_dy_inner_v = _mm_div_ps(xmm_dy_inner_v, xmm_norm1);
  
        _mm_store_ps(&partialDx[i][lastBlock], xmm_dx_inner_v);
        _mm_store_ps(&partialDy[i + 1][lastBlock], xmm_dy_inner_v);
        //printf("e-i: %d, j: %d\n", i, j);

        /* 
        for(j = 0; j < width; j++){
          imagecopy = partialImageCopy[i][j];
          dx_v = partialDx[i][j];
          dy_v = partialDy[i + 1][j];

          //need to deal with last element
          if(j < width - 1){
            dx_inner_v = (partialImageCopy[i][j + 1] - imagecopy) * fSigma;
          }else{
            dx_inner_v = 0.0;
          }
          //need to deal with last line in rank (n-1)
          if(rank == (num_tasks - 1) && (i == (partialHeight - 1))){
            dy_inner_v = 0.0;
          }else{
            dy_inner_v = (partialImageCopy[i + 1][j] - imagecopy) * fSigma;
          }

          dx_v += dx_inner_v; 
          dy_v += dy_inner_v;

          norm = max(sqrt(dx_v * dx_v + dy_v * dy_v), 1.0);
          dx_v /= norm; 
          dy_v /= norm; 
          //P = project_nd( P + sigma * nabla(X), 1.0 )
          //
          partialDx[i][j] = dx_v;
          partialDy[i + 1][j] = dy_v;
        }*/
      }

      if(rank > 0){
        MPI_Recv(partialDy[0], width, MPI_FLOAT, rank - 1, 126, MPI_COMM_WORLD, &status);
      }
      if(rank < num_tasks - 1){
        MPI_Send(partialDy[partialHeight], width, MPI_FLOAT, rank + 1, 126, MPI_COMM_WORLD);
      }  
      MPI_Barrier(MPI_COMM_WORLD);

            
      for(i = 0; i < partialHeight; i++){
        __m128 xmm_temp1, xmm_temp2, xmm_x1, xmm_imagecopy;
        float* dx_ptr = partialDx[i];
        float* dy_ptr = partialDy[i + 1];
        float* shift_dx_ptr = &(partialDx[i][3]);//shift 3 pixels since first round we do not need to use this 3 pixels
        float* shift_dy_ptr = partialDy[i];
        float* imagecopy_ptr = partialImageCopy[i];
        float* data_ptr = partialData[i];
        
        for(j = 0; j < width / 4; j++){
          __m128 xmm_nt = xmm_zero_mask;
          if(j == 0){//first line
            xmm_temp1 = _mm_loadu_ps(dx_ptr); 
            xmm_temp2 = _mm_setr_ps(0.0, partialDx[i][0], partialDx[i][1], partialDx[i][2]);
            //if(i == 0){
            //  printf("dx: %f, pDx: %f\n", dx_ptr[1], partialDx[i][0]);
            //}
          }else if(j == (width / 4 - 1)){//last line
            xmm_temp1 = _mm_setr_ps(dx_ptr[0], dx_ptr[1], dx_ptr[2], 0.0);
            xmm_temp2 = _mm_loadu_ps(shift_dx_ptr); 
          }else{
            xmm_temp1 = _mm_loadu_ps(dx_ptr); 
            xmm_temp2 = _mm_loadu_ps(shift_dx_ptr); 
            shift_dx_ptr += 4;
          }

          xmm_nt = _mm_sub_ps(xmm_nt, xmm_temp1);
          xmm_nt = _mm_add_ps(xmm_nt, xmm_temp2);
          /*if(i == 0 && j == 0){
             float test[4];
            _mm_store_ps(&test, xmm_temp1);
            printf("tmp10: %f\n", test[0]);
            printf("tmp11: %f\n", test[1]);
            printf("tmp12: %f\n", test[2]);
            printf("tmp13: %f\n", test[3]);
            _mm_store_ps(&test, xmm_temp2);
            printf("tmp20: %f\n", test[0]);
            printf("tmp21: %f\n", test[1]);
            printf("tmp22: %f\n", test[2]);
            printf("tmp23: %f\n", test[3]);
            _mm_store_ps(&test, xmm_nt);
            printf("nt0: %f\n", test[0]);
            printf("nt1: %f\n", test[1]);
            printf("nt2: %f\n", test[2]);
            printf("nt3: %f\n", test[3]);
          }*/

          xmm_temp1 = xmm_zero_mask;
          xmm_temp2 = xmm_zero_mask;

          if((rank == 0) && (i == 0)){//first line
            xmm_temp1 = _mm_loadu_ps(dy_ptr);
          //last line
          }else if((rank == (num_tasks - 1)) && (i == (partialHeight - 1))){
            xmm_temp2 = _mm_loadu_ps(shift_dy_ptr);
          //others
          }else{
            xmm_temp1 = _mm_loadu_ps(dy_ptr);
            xmm_temp2 = _mm_loadu_ps(shift_dy_ptr);
          }

          xmm_nt = _mm_sub_ps(xmm_nt, xmm_temp1);
          xmm_nt = _mm_add_ps(xmm_nt, xmm_temp2);

          xmm_temp1 = _mm_loadu_ps(imagecopy_ptr);
          xmm_temp2 = _mm_loadu_ps(data_ptr);
          xmm_nt = _mm_mul_ps(xmm_nt, xmm_tau_mask);
          xmm_nt = _mm_sub_ps(xmm_temp1, xmm_nt);
          xmm_temp2 = _mm_sub_ps(xmm_temp2, xmm_nt);
          xmm_temp2 = _mm_min_ps(xmm_temp2, xmm_shrink1_mask);
          xmm_temp2 = _mm_max_ps(xmm_temp2, xmm_shrink2_mask);
          xmm_temp2 = _mm_add_ps(xmm_temp2, xmm_nt);

          xmm_temp1 = _mm_sub_ps(xmm_temp2, xmm_temp1);
          xmm_temp1 = _mm_mul_ps(xmm_temp1, xmm_theta_mask);
          xmm_temp2 = _mm_add_ps(xmm_temp2, xmm_temp1); 

          _mm_store_ps(&partialImageCopy[i][j * 4], xmm_temp2);

          dx_ptr += 4;
          dy_ptr += 4;
          shift_dy_ptr += 4;
          imagecopy_ptr += 4;
          data_ptr += 4;
        }
      }
    }

    //copy final result back
    //rearrange local 2D data to local 1D data
    counter = 0;
    for(i = 0; i < partialHeight; i++){
      for(j = 0; j < width; j++){
        localData1D[counter] = partialImageCopy[i][j];
        counter++;
      }
    }

    //send back to rank 0
    MPI_Gather(localData1D, partialHeight * width, MPI_FLOAT, data1D, partialHeight * width, MPI_FLOAT, 0, MPI_COMM_WORLD);

    //rearrange 1D data to 2D data
    if(0 == rank){
      int k = 0;
      for(i = 0; i < height; i++){
        for(j = 0; j < width; j++){
          data[i][j] = data1D[k];
          k++; 
        }
      } 
    }

    //release resources
    for(i = 0; i < partialHeight + 1; i++){
      free(partialData[i]);
      free(partialDy[i]);
      free(partialImageCopy[i]);
    }
    for(i = 0; i < partialHeight; i++){
      free(partialDx[i]);
    }
     
    free(fTempMask);
    free(partialData);
    free(partialDx);
    free(partialDy);
    free(partialImageCopy);
  }
 
  if(rank == 0){ 
    get_timestamp(&tsFinish);
    elapsed = timestamp_diff_in_seconds(tsStart, tsFinish);
    printf("Time elapsed for ROF MPI+SSE solver is %f seconds.\n", elapsed);
  }

  //save to file
  if(rank == 0){
    FILE *outputFile = fopen("../tvl1SSEResultFile.txt", "w+");
    fprintf(outputFile, "%d\n", height);
    fprintf(outputFile, "%d\n", width);
    for(i = 0; i < height; i++){
      for(j = 0; j < width; j++){
        fprintf(outputFile, "%f\n", data[i][j]);
      }
    }
    fclose(outputFile);

    //release resources
    for(i = 0; i < height; i++){
      free(data[i]);
    }
    free(data);
  }

 
  MPI_Finalize();
  return 0;
}
