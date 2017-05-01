#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <string.h>
#include "solver.h"

#define ITER 101

/* calculate nabla 
 * 
 * input
 * ptr       : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * dx        : differential x
 * dy        : differential y
 * output
 * dx        : differential x
 * dy        : differential y
*/

void nabla(double** ptr, int width, int height, double** dx, double** dy){
  //x part
  for(int i = 0 ; i < height; i++){
    for(int j = 0 ; j < width - 1; j++){
      dx[i][j] = ptr[i][j + 1] - ptr[i][j]; 
    }
  }
  //y part
  for(int i = 0 ; i < height - 1; i++){
    for(int j = 0 ; j < width; j++){
      dy[i][j] = ptr[i + 1][j] - ptr[i][j]; 
    }
  } 
}

/* calculate nabla 
 * 
 * input
 * dx        : dx matrix pointer
 * dy        : dy matrix pointer
 * width     : matrix width
 * height    : matrix height
 * output
 * ptr       : reverse pointer
*/

void nablaT(double** dx, double** dy, int width, int height, double** ptr){
  for(int i = 0; i < height; i++){
    memset(ptr[i], 0.0, width * sizeof(double));
  }
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width - 1; j++){
      ptr[i][j] -= dx[i][j];
    }
  } 
  for(int i = 0; i < height; i++){
    for(int j = 1; j < width; j++){
      ptr[i][j] += dx[i][j - 1];
    }
  } 
  for(int i = 0; i < height - 1; i++){
    for(int j = 0; j < width; j++){
      ptr[i][j] -= dy[i][j];
    }
  } 
  for(int i = 1; i < height; i++){
    for(int j = 0; j < width; j++){
      ptr[i][j] += dy[i - 1][j];
    }
  } 
}

/* multiply factor to each of matrix's elements 
 * 
 * input
 * ptr       : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * factor    : factor to multiplication
 * output
 * result    : matrix pointer
*/

void matrix_mul(double** ptr, int width, int height, double factor, double** result){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      result[i][j] = ptr[i][j] * factor;
    }
  }
}

/* add matrix1's elements to matrix2's elements
 *
 * input
 * ptr1      : matrix pointer
 * ptr2      : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * output
 * ptr1      : matrix pointer
*/

void matrix_add(double** ptr1, double** ptr2, int width, int height){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      ptr1[i][j] += ptr2[i][j];
    }
  }
}

/* minus matrix1's elements by matrix2's elements
 *
 * input
 * ptr1      : matrix pointer
 * ptr2      : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * output
 * result      : matrix pointer
*/

void matrix_minus(double** ptr1, double** ptr2, int width, int height, double** result){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      result[i][j] = ptr1[i][j] - ptr2[i][j];
    }
  }
}

/* divide matrix1's elements by matrix2's elements
 *
 * input
 * ptr1      : matrix pointer
 * ptr2      : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * output
 * ptr1      : matrix pointer
*/

void matrix_divide(double** ptr1, double** ptr2, int width, int height){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      ptr1[i][j] /= ptr2[i][j];
    }
  }
}

/* divide matrix1's elements by factor
 *
 * input
 * ptr1      : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * output
 * ptr1      : matrix pointer
*/

void matrix_divide(double** ptr1, int width, int height, double factor){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      ptr1[i][j] /= factor;
    }
  }
}

/* calculate matrix1's elements and matrix2's elements L2 norm
 *
 * input
 * ptr1      : matrix pointer
 * ptr2      : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * output
 * norm      : matrix pointer
*/

void anorm(double** ptr1, double** ptr2, int width, int height, double** norm){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      norm[i][j] = sqrt(ptr1[i][j] * ptr1[i][j] + ptr2[i][j] * ptr2[i][j]);
    }
  }
}

/* project 
 *
 * input
 * ptr1      : matrix pointer
 * ptr2      : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * r         : factor
 * output
*/

void project_nd(double** ptr1, double** ptr2, int width, int height, double r){
  
  double** norm = new double*[height];
  for(int i = 0; i < height; i++){
    norm[i] = new double[width];
  }
 
  anorm(ptr1, ptr2, width, height, norm);
  //matrix_mul(norm, width, height, 1.0);
  matrix_max(norm, width, height, 1.0); 
  matrix_divide(ptr1, norm, width, height); 
  matrix_divide(ptr2, norm, width, height); 
  
  for(int i = 0; i < height; i++){
    delete [] norm[i];
  }
  delete [] norm;
}

/* matrix maximum compared to a number
 *
 * input
 * ptr       : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * r         : factor
 * output
 * ptr       : matrix pointer
*/

void matrix_max(double** ptr, int width, int height, double maximum){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      ptr[i][j] = std::max(maximum, ptr[i][j]);
    }
  }
}

/* matrix value bound in minV and maxV
 *
 * input
 * ptr       : matrix pointer
 * width     : matrix width
 * height    : matrix height
 * minV      : min value
 * maxV      : max value
 * output
 * ptr       : matrix pointer
*/

void clip(double** ptr, int width, int height, double minV, double maxV){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      ptr[i][j] = std::max(std::min(maxV, ptr[i][j]), minV);
    }
  }
}

void solve_rof(double** ptr, int width, int height, int channels, double** result, double clambda){
  double fL2 = 8.0;
  double fTau = 0.02; 
  double fSigma = 1.0 / (fL2 * fTau);  
  double fTheta = 1.0;
  int iteration = ITER;

  double** dx = new double*[height];
  double** dy = new double*[height];
  double** dx_inner = new double*[height];
  double** dy_inner = new double*[height];
  double** nabla_t = new double*[height];
  double** temp = new double*[height];
  double** imageCopy = new double*[height];
  
  for(int i = 0; i < height; i++){
    dx[i] = new double[width];
    dy[i] = new double[width];
    dx_inner[i] = new double[width];
    dy_inner[i] = new double[width];
    nabla_t[i] = new double[width];
    temp[i] = new double[width];
    imageCopy[i] = new double[width];
    memcpy(imageCopy[i], ptr[i], width * sizeof(double));
    for(int j = 0; j < width; j++){
      dx[i][j] = 0.0;
      dy[i][j] = 0.0;
      dx_inner[i][j] = 0.0;
      dy_inner[i][j] = 0.0;
      nabla_t[i][j] = 0.0;
      temp[i][j] = 0.0;
    }
  }
  //X = img.copy()
  nabla(ptr, width, height, dx, dy);
  //P = nabla(X)

  for(int i = 0; i < iteration; i++){
    nabla(imageCopy, width, height, dx_inner, dy_inner);
    matrix_mul(dx_inner, width, height, fSigma, dx_inner);
    matrix_mul(dy_inner, width, height, fSigma, dy_inner);
    matrix_add(dx, dx_inner, width, height);
    matrix_add(dy, dy_inner, width, height);
    project_nd(dx, dy, width, height, 1.0);
    //P = project_nd( P + sigma * nabla(X), 1.0 )
   
    double lt = clambda * fTau;
    //lt = clambda * tau
    
    nablaT(dx, dy, width, height, nabla_t);
    matrix_mul(nabla_t, width, height, fTau, nabla_t);
    matrix_mul(ptr, width, height, lt, temp);
    matrix_minus(imageCopy, nabla_t, width, height, nabla_t);
    matrix_add(temp, nabla_t, width, height);
    matrix_divide(temp, width, height, 1.0 + lt);
    //X1 = (X - tau * nablaT(P) + lt * img) / (1.0 + lt)
    
    matrix_minus(temp, imageCopy, width, height, imageCopy);
    matrix_add(imageCopy, temp, width, height);
    //X = X1 + theta * (X1 - X)
    
    //if i % 10 == 0:
      //print "%.2f" % calc_energy_ROF(X, img, clambda),
  
  }

  //copy final result back
  for(int i = 0; i < height; i++){
    memcpy(result[i], imageCopy[i], width * sizeof(double));
  }

  //release resources
  for(int i = 0; i < height; i++){
    delete [] dx[i];
    delete [] dy[i];
    delete [] dx_inner[i];
    delete [] dy_inner[i];
    delete [] nabla_t[i];
    delete [] temp[i];
    delete [] imageCopy[i];
  }

  delete [] dx; 
  delete [] dy; 
  delete [] dx_inner; 
  delete [] dy_inner; 
  delete [] nabla_t;
  delete [] temp;
  delete [] imageCopy;
}

void solve_tvl1(double** ptr, int width, int height, int channels, double** result, double clambda){
  double fL2 = 8.0;
  double fTau = 0.02; 
  double fSigma = 1.0 / (fL2 * fTau);  
  double fTheta = 1.0;
  int iteration = ITER;

  double** dx = new double*[height];
  double** dy = new double*[height];
  double** dx_inner = new double*[height];
  double** dy_inner = new double*[height];
  double** nabla_t = new double*[height];
  double** temp = new double*[height];
  double** imageCopy = new double*[height];
  
  for(int i = 0; i < height; i++){
    dx[i] = new double[width];
    dy[i] = new double[width];
    dx_inner[i] = new double[width];
    dy_inner[i] = new double[width];
    nabla_t[i] = new double[width];
    temp[i] = new double[width];
    imageCopy[i] = new double[width];
    memcpy(imageCopy[i], ptr[i], width * sizeof(double));
    for(int j = 0; j < width; j++){
      dx[i][j] = 0.0;
      dy[i][j] = 0.0;
      dx_inner[i][j] = 0.0;
      dy_inner[i][j] = 0.0;
      nabla_t[i][j] = 0.0;
      temp[i][j] = 0.0;
    }
  }
  //X = img.copy()
  nabla(ptr, width, height, dx, dy);
  //P = nabla(X)

  for(int i = 0; i < iteration; i++){
    nabla(imageCopy, width, height, dx_inner, dy_inner);
    matrix_mul(dx_inner, width, height, fSigma, dx_inner);
    matrix_mul(dy_inner, width, height, fSigma, dy_inner);
    matrix_add(dx, dx_inner, width, height);
    matrix_add(dy, dy_inner, width, height);
    project_nd(dx, dy, width, height, 1.0);
    //P = project_nd( P + sigma*nabla(X), 1.0 )
   
    nablaT(dx, dy, width, height, nabla_t);
    matrix_mul(nabla_t, width, height, fTau, nabla_t);
    matrix_minus(imageCopy, nabla_t, width, height, nabla_t);
    matrix_minus(ptr, nabla_t, width, height, temp);
    clip(temp, width, height, -clambda * fTau, clambda * fTau);
    matrix_add(temp, nabla_t, width, height);
    //X1 = shrink_1d(X - tau*nablaT(P), img, clambda*tau)
    //def shrink_1d(X, F, step):
    //return X + np.clip(F - X, -step, step)
  
    matrix_minus(temp, imageCopy, width, height, imageCopy);
    matrix_add(imageCopy, temp, width, height);
    //X = X1 + theta * (X1 - X)
    //  if i % 10 == 0:
    //        print "%.2f" % calc_energy_TVL1(X, img, clambda),
  }

  //copy final result back
  for(int i = 0; i < height; i++){
    memcpy(result[i], imageCopy[i], width * sizeof(double));
  }

  //release resources
  for(int i = 0; i < height; i++){
    delete [] dx[i];
    delete [] dy[i];
    delete [] dx_inner[i];
    delete [] dy_inner[i];
    delete [] nabla_t[i];
    delete [] temp[i];
    delete [] imageCopy[i];
  }

  delete [] dx; 
  delete [] dy; 
  delete [] dx_inner; 
  delete [] dy_inner; 
  delete [] nabla_t;
  delete [] temp;
  delete [] imageCopy;
}

void solve_rof_arranged(double** ptr, int width, int height, int channels, double** result, double clambda){
  double fL2 = 8.0;
  double fTau = 0.02; 
  double fSigma = 1.0 / (fL2 * fTau);  
  double fTheta = 1.0;
  int iteration = ITER;
  double norm;
  double dx_v, dy_v, dx_inner_v, dy_inner_v, nt, x1, imagecopy;

  double** dx = new double*[height];
  double** dy = new double*[height];
  double** imageCopy = new double*[height];
  
  for(int i = 0; i < height; i++){
    dx[i] = new double[width];
    dy[i] = new double[width];
    imageCopy[i] = new double[width];
    memcpy(imageCopy[i], ptr[i], width * sizeof(double));
    for(int j = 0; j < width; j++){
      dx[i][j] = 0.0;
      dy[i][j] = 0.0;
    }
  }
  //X = img.copy()
  //x part
  for(int i = 0 ; i < height; i++){
    for(int j = 0 ; j < width - 1; j++){
      dx[i][j] = ptr[i][j + 1] - ptr[i][j]; 
    }
  }
  //y part
  for(int i = 0 ; i < height - 1; i++){
    for(int j = 0 ; j < width; j++){
      dy[i][j] = ptr[i + 1][j] - ptr[i][j]; 
    }
  } 
  //P = nabla(X)

  double lt = clambda * fTau;
  //lt = clambda * tau
  
  for(int k = 0; k < iteration; k++){
    for(int i = 0; i < height; i++){
      for(int j = 0 ; j < width; j++){
        imagecopy = imageCopy[i][j];
        dx_v = dx[i][j];
        dy_v = dy[i][j];

        if(j < width - 1){
          dx_inner_v = (imageCopy[i][j + 1] - imagecopy) * fSigma;
        }else{
          dx_inner_v = 0.0;
        }
        if(i < height - 1){
          dy_inner_v = (imageCopy[i + 1][j] - imagecopy) * fSigma;
        }else{
          dy_inner_v = 0.0;
        }

        dx_v += dx_inner_v; 
        dy_v += dy_inner_v;
        norm = std::max(sqrt(dx_v * dx_v + dy_v * dy_v), 1.0);
        dx_v /= norm; 
        dy_v /= norm;

        //P = project_nd( P + sigma * nabla(X), 1.0 )
        //
        dx[i][j] = dx_v;
        dy[i][j] = dy_v;

        nt = 0.0;
        if(j < width - 1){
          nt -= dx_v;
        }
        if(j >= 1){
          nt += dx[i][j - 1];
        }
        if(i < height - 1){
          nt -= dy_v;
        }
        if(i >= 1){
          nt += dy[i - 1][j];
        }

        x1 = (imagecopy - nt * fTau + lt * ptr[i][j]) / (1.0 + lt);
        imageCopy[i][j] = x1 + fTheta * (x1 - imagecopy);
      }
    }
    //if i % 10 == 0:
      //print "%.2f" % calc_energy_ROF(X, img, clambda),
  }

  //copy final result back
  for(int i = 0; i < height; i++){
    memcpy(result[i], imageCopy[i], width * sizeof(double));
  }

  //release resources
  for(int i = 0; i < height; i++){
    delete [] dx[i];
    delete [] dy[i];
    delete [] imageCopy[i];
  }

  delete [] dx; 
  delete [] dy; 
  delete [] imageCopy;
 
}

void solve_tvl1_arranged(double** ptr, int width, int height, int channels, double** result, double clambda){

  double fL2 = 8.0;
  double fTau = 0.02; 
  double fSigma = 1.0 / (fL2 * fTau);  
  double fTheta = 1.0;
  int iteration = ITER;
  double norm;
  double dx_v, dy_v, dx_inner_v, dy_inner_v, nt, x1, x, imagecopy;

  double** dx = new double*[height];
  double** dy = new double*[height];
  double** imageCopy = new double*[height];
  
  for(int i = 0; i < height; i++){
    dx[i] = new double[width];
    dy[i] = new double[width];
    imageCopy[i] = new double[width];
    memcpy(imageCopy[i], ptr[i], width * sizeof(double));
    for(int j = 0; j < width; j++){
      dx[i][j] = 0.0;
      dy[i][j] = 0.0;
    }
  }
  //X = img.copy()
  //x part
  for(int i = 0 ; i < height; i++){
    for(int j = 0 ; j < width - 1; j++){
      dx[i][j] = ptr[i][j + 1] - ptr[i][j]; 
    }
  }
  //y part
  for(int i = 0 ; i < height - 1; i++){
    for(int j = 0 ; j < width; j++){
      dy[i][j] = ptr[i + 1][j] - ptr[i][j]; 
    }
  } 
  //P = nabla(X)

  double shrink = clambda * fTau;
  //shrink 
  for(int k = 0; k < iteration; k++){
    for(int i = 0; i < height; i++){
      for(int j = 0 ; j < width; j++){
        imagecopy = imageCopy[i][j];
        dx_v = dx[i][j];
        dy_v = dy[i][j];
        
        if(j < width - 1){
          dx_inner_v = (imageCopy[i][j + 1] - imagecopy) * fSigma;
        }else{
          dx_inner_v = 0.0;
        }
        if(i < height - 1){
          dy_inner_v = (imageCopy[i + 1][j] - imagecopy) * fSigma;
        }else{
          dy_inner_v = 0.0;
        }

        dx_v += dx_inner_v; 
        dy_v += dy_inner_v;
        norm = std::max(sqrt(dx_v * dx_v + dy_v * dy_v), 1.0);
        dx_v /= norm; 
        dy_v /= norm; 
        
        //P = project_nd( P + sigma * nabla(X), 1.0 )
        
        dx[i][j] = dx_v;
        dy[i][j] = dy_v;

        nt = 0.0;
        if(j < width - 1){
          nt -= dx_v;
        }
        if(j >= 1){
          nt += dx[i][j - 1];
        }
        if(i < height - 1){
          nt -= dy_v;
        }
        if(i >= 1){
          nt += dy[i - 1][j];
        }

        x = imagecopy - nt * fTau;
        x1 = x + std::max(std::min(ptr[i][j] - x, shrink), -shrink);
        imageCopy[i][j] = x1 + fTheta * (x1 - imagecopy);
      }
    }
    //if i % 10 == 0:
      //print "%.2f" % calc_energy_ROF(X, img, clambda),
  }

  //copy final result back
  for(int i = 0; i < height; i++){
    memcpy(result[i], imageCopy[i], width * sizeof(double));
  }

  //release resources
  for(int i = 0; i < height; i++){
    delete [] dx[i];
    delete [] dy[i];
    delete [] imageCopy[i];
  }

  delete [] dx; 
  delete [] dy; 
  delete [] imageCopy;
 
}

