#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include "../utility/util.h"

#ifndef max
  #define max(a,b) ((a) > (b) ? (a) : (b))
#endif

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
    data = (float **) calloc(sizeof(float*), height);
    for(i = 0; i < height; i++){
     data[i] = (float *) calloc(sizeof(float), width);
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
    data1D = (float *) calloc(sizeof(float), width * height);
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
  float *localData1D = (float *) calloc(sizeof(float), width * (partialHeight + 1));

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
    float clambda = 8.0;
    float fL2 = 8.0;
    float fTau = 0.02; 
    float fSigma = 1.0 / (fL2 * fTau);  
    float fTheta = 1.0;
    int iteration = ITER;
    float norm;
    float dx_v, dy_v, dx_inner_v, dy_inner_v, nt, x1, imagecopy;
    float **partialData = (float **) calloc(sizeof(float*), partialHeight + 1);
    float **partialDx = (float **) calloc(sizeof(float*), partialHeight);
    float **partialDy = (float **) calloc(sizeof(float*), partialHeight + 1);
    float **partialImageCopy = (float **) calloc(sizeof(float*), partialHeight + 1);
    for(i = 0; i < partialHeight + 1; i++){
      partialData[i] = (float *) calloc(sizeof(float), width);
      partialDy[i] = (float *) calloc(sizeof(float), width);
      partialImageCopy[i] = (float *) calloc(sizeof(float), width);
    }
    for(i = 0; i < partialHeight; i++){
      partialDx[i] = (float *) calloc(sizeof(float), width);
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
    //lt = clambda * tau
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
        for(j = 0; j < width; j++){
          imagecopy = partialImageCopy[i][j];
          dx_v = partialDx[i][j];
          dy_v = partialDy[i + 1][j];
          /*if(i == 0 && j == 0){
            printf("dx_v: %f\n", dx_v);
            printf("dy_v: %f\n", dy_v);
          }*/

          if(j < width - 1){
            dx_inner_v = (partialImageCopy[i][j + 1] - imagecopy) * fSigma;
          }else{
            dx_inner_v = 0.0;
          }
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
        }
      }

      if(rank > 0){
        MPI_Recv(partialDy[0], width, MPI_FLOAT, rank - 1, 126, MPI_COMM_WORLD, &status);
      }
      if(rank < num_tasks - 1){
        MPI_Send(partialDy[partialHeight], width, MPI_FLOAT, rank + 1, 126, MPI_COMM_WORLD);
      }  
      MPI_Barrier(MPI_COMM_WORLD);

      for(i = 0; i < partialHeight; i++){
        for(j = 0; j < width; j++){
          nt = 0.0;
          dx_v = partialDx[i][j];
          dy_v = partialDy[i + 1][j];
          imagecopy = partialImageCopy[i][j];
          if(j < width - 1){
            nt -= dx_v;
          }
          if(j >= 1){
            nt += partialDx[i][j - 1];
          }/*
          if(i == 0 && j == 1){
            printf("dx_v: %f, pDx: %f\n", dx_v, partialDx[i][j - 1]);
          }
          if(i == 0 && j == 0){
            printf("nt0: %f\n", nt);
          }
          if(i == 0 && j == 1){
            printf("nt1: %f\n", nt);
          }
          if(i == 0 && j == 2){
            printf("nt2: %f\n", nt);
          }
          if(i == 0 && j == 3){
            printf("nt3: %f\n", nt);
          }*/

          if((rank == 0) && (i == 0)){
            nt -= dy_v;
          }else if((rank == (num_tasks - 1)) && (i == (partialHeight - 1))){
            nt += partialDy[i][j];
          }else{
            nt -= dy_v;
            nt += partialDy[i][j];
          }/*
          if(i == 0 && j == 0){
            printf("nt0: %f\n", nt);
          }
          if(i == 0 && j == 1){
            printf("nt1: %f\n", nt);
          }
          if(i == 0 && j == 2){
            printf("nt2: %f\n", nt);
          }
          if(i == 0 && j == 3){
            printf("nt3: %f\n", nt);
          }*/

          x1 = (imagecopy - nt * fTau + lt * partialData[i][j]) / (1.0 + lt);
          partialImageCopy[i][j] = x1 + fTheta * (x1 - imagecopy);
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

    free(partialData);
    free(partialDx);
    free(partialDy);
    free(partialImageCopy);
  }
 
  if(rank == 0){ 
    get_timestamp(&tsFinish);
    elapsed = timestamp_diff_in_seconds(tsStart, tsFinish);
    printf("Time elapsed for ROF MPI solver is %f seconds.\n", elapsed);
  }

  //save to file
  if(rank == 0){
    FILE *outputFile = fopen("../rofResultFile.txt", "w+");
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
