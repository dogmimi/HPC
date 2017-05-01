#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <string>
#include "imageutility.h"

using namespace cv;
using namespace std;

//if file is not existed or any error happened, return false
bool getImageInfo(string fileName, int& width, int& height, int& channels, int& depth){
  Mat image;
  image = imread(fileName, 1);
  if(image.empty()){
    return false;
  }

  height = image.rows;
  width = image.cols;
  depth = image.depth();
  channels = image.channels();

  return true; 
}

//if file is not existed or any error happened, return false
bool readImageFromFile(double** imagePtr, string fileName){
  Mat image;
  image = imread(fileName, 1);
  if(image.empty()){
    return false;
  }

  int height = image.rows;
  int width = image.cols;
  int depth = image.depth();
  int channels = image.channels();

  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "originalFile.txt");
  fd = fopen(filename, "w+");
 
  fprintf(fd, "%d\n", height);
  fprintf(fd, "%d\n", width);

  unsigned char* ptr = (unsigned char*)image.data;
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      //int gray = (ptr[0] + ptr[1] + ptr[2]) / 3;
      int gray = (ptr[0] * 114 + ptr[1] * 587 + ptr[2] * 299) / 1000;
        fprintf(fd, "%d\n", gray);
      imagePtr[i][j] = (double)gray / 255.0;
      ptr += 3;
    }
  }

  fclose(fd);

  return true; 
}

void saveImageToFile(double** imagePtr, int width, int height, string fileName){
  //normalization
  double dMin = 1.0;
  double dMax = 0.0;
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      if(imagePtr[i][j] > dMax){
        dMax = imagePtr[i][j];
      } 
      if(imagePtr[i][j] < dMin){
        dMin = imagePtr[i][j];
      } 
    }
  }

  //create mat matrix and assign value
  Mat output_image = Mat(height, width, CV_64FC1);
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      output_image.at<double>(i, j) = std::min(std::max((imagePtr[i][j] - dMin) / (dMax - dMin) * 255.0, 0.0), 255.0);
    }
  }
  
  imwrite(fileName, output_image);
}

void ReadFileTo2DArray(string fileName, double** data){
  int i, j, width, height;
  float v;
  FILE *inputFile;
  inputFile = fopen(fileName.c_str(), "r");
  fscanf(inputFile, "%d", &height);
  fscanf(inputFile, "%d", &width);
  for(i = 0; i < height; i++){
    for(j = 0; j < width; j++){
      fscanf(inputFile, "%f", &v);
      data[i][j] = v;
    }
  }
  fclose(inputFile);
}
