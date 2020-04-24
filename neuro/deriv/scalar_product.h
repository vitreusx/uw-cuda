//
//  scalar_product.h
//  
//
//  Created by Witold Rudnicki on 09.03.2020.
//

#ifndef scalar_product_h
#define scalar_product_h
#include <cuda_runtime_api.h>
using namespace std;
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define real float
#define BlockSize 1024

#endif /* scalar_product_h */

int ReadCSV(char* CSVfile,float** CNV);
float scalar(float *x, float* y, int len);
void normalize(float** sourceMat,float**  destMat, int Size, int Len);
void similarity(float** sourceMat, float* simMat, int Size, int Len);

double scalar_d(float *x, float* y, int len);
void normalize_d(float** sourceMat,float**  destMat, int Size, int Len);
__global__ void scalar_2(float *Mat, int ind_x, int ind_y, int Len, int Size, float *resMat );
__global__ void scalar_1(float *Mat, int ind_x, int ind_y, int Len, int Size, float *resMat );
void similarity_gpu(float** sourceMat, float* simMat, int Size, int Len);
float correlation(float *x,float *y,int Size);
