//
//  Mandelbrot_cuda.h
//  Created by Witold Rudnicki 
//
//

#ifndef ____Mandelbrot__
#define ____Mandelbrot__

using namespace std;
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream>

#endif /* defined(____Mandelbrot__) */

void computeMandelbrot_2D(float X0, float Y0, float X1, float Y1, int POZ, int PION, int ITER, int *Mandel);
void computeMandelbrot_1D(float X0, float Y0, float X1, float Y1, int POZ, int PION, int ITER, int *Mandel);
void makePicture(int *Mandel, int width, int height, int MAX);
void makePictureInt(int *Mandel,int width, int height, int MAX);
__global__ void cudaMandelbrot_1D(float X0, float Y0, float X1, float Y1, int POZ, int PION, int ITER,int *Mandel);

__global__ void cudaMandelbrot_2D(float X0, float Y0, float X1, float Y1, int POZ, int PION, int ITER,int *Mandel);


