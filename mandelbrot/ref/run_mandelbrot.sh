#!/bin/bash
g++ -o mandelbrot_cpu Mandelbrot.cpp
./mandelbrot_cpu -1. -1. 1. 1. 3000 3000 256 > cpu.out
mv Mandel.ppm Mandel_cpu.ppm
nvcc -o mandelbrot_gpu Mandelbrot.cu 
./mandelbrot_gpu -1. -1. 1. 1. 3000 3000 256 > gpu.out
mv Mandel.ppm Mandel_gpu.ppm 
cmp Mandel_gpu.ppm Mandel_cpu.ppm > cmp.out 



