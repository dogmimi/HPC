#ifndef IMAGEUTILITY_H
#define IMAGEUTILITY_H

#include <string>
using namespace std;

bool getImageInfo(string fileName, int& width, int& height, int& channels, int& depth);
bool readImageFromFile(double** imagePtr, string fileName);
void saveImageToFile(double** imagePtr, int width, int height, string fileName); 
void ReadFileTo2DArray(string fileName, double** data);

#endif
