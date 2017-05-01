#ifndef SOLVER_H
#define SOLVER_H

//original implementation and related functions
void solve_rof(double** ptr, int width, int height, int channels, double** result, double clambda = 8.0);
void solve_tvl1(double** ptr, int width, int height, int channels, double** result, double clambda = 1.0);
void nabla(double** ptr, int width, int height, double** dx, double** dy);
void nablaT(double** dx, double** dy, int width, int height, double** ptr);
void matrix_mul(double** ptr, int width, int height, double factor, double** result);
void matrix_add(double** ptr1, double** ptr2, int width, int height);
void matrix_minus(double** ptr1, double** ptr2, int width, int height, double** result);
void matrix_divide(double** ptr1, double** ptr2, int width, int height);
void matrix_divide(double** ptr1, int width, int height, double factor);
void matrix_max(double** ptr, int width, int height, double maximum);
void anorm(double** ptr1, double** ptr2, int width, int height, double** norm);
void project_nd(double** ptr1, double** ptr2, int width, int height, double r);
void clip(double** ptr, int width, int height, double minV, double maxV);

//refactor all related function to fit in for loop for easily parallelization
void solve_rof_arranged(double** ptr, int width, int height, int channels, double** result, double clambda = 8.0);
void solve_tvl1_arranged(double** ptr, int width, int height, int channels, double** result, double clambda = 1.0);

#endif
