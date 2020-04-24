//
//  Mandelbrot.cpp
//  
//
//  Created by Witold Rudnicki
//
//

#include "Mandelbrot_cuda.h"

__global__
void cudaMandelbrot_1D(float X0, float Y0, float X1, float Y1, int POZ, int PION, int ITER,int *Mandel ){
    float    dX=(X1-X0)/(POZ-1);
    float    dY=(Y1-Y0)/(PION-1);
    float x,y,Zx,Zy,tZx;
    int i;
    int SIZE=POZ*PION;
    int pion, poz;
    
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if (indx >= SIZE) return;

    pion=indx / POZ;
    poz=indx % POZ;
    x=X0+poz*dX;
    y=Y0+pion*dY;
    Zx=x;
    Zy=y;
    i=0;
            
    while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) ){
        tZx = Zx*Zx-Zy*Zy+x;
        Zy = 2*Zx*Zy+y;
        Zx = tZx;
            
        i++;
    }
    Mandel[indx] = i;
}

__global__
void cudaMandelbrot_2D(float X0, float Y0, float X1, float Y1, int POZ, int PION, int ITER,int *Mandel ){
    float    dX=(X1-X0)/(POZ-1);
    float    dY=(Y1-Y0)/(PION-1);
    float x,y,Zx,Zy,tZx;
    int i;
    
    int poz = blockIdx.x * blockDim.x + threadIdx.x;
    if (poz >= POZ) return;

    int pion = blockIdx.y * blockDim.y + threadIdx.y;
    if (pion >= PION) return;

    x=X0+poz*dX;
    y=Y0+pion*dY;
    Zx=x;
    Zy=y;
    i=0;
    while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) )
    {
        tZx = Zx*Zx-Zy*Zy+x;
        Zy = 2*Zx*Zy+y;
        Zx = tZx;
    
        i++;
    }
    Mandel[pion*POZ+poz] = i;
}

int main(int argc, char **argv) {
    if (argc!=8) {
        printf("Wywołanie %s LD_Re, LD_Im, PG_Re, PG_Im, Poziom, Pion, Iteracje\n ",argv[0]);
        exit(1);
    }
    //Ustaw obszar obliczeń
    //{X0,Y0} - lewy dolny róg
    float X0=atof(argv[1]);
    float Y0=atof(argv[2]);

    //{X1,Y1} - prawy górny róg
    float X1=atof(argv[3]);
    float Y1=atof(argv[4]);

    //Ustal rozmiar w pikselach
    //{POZ,PION}
    int POZ=atoi(argv[5]);
    int PION=atoi(argv[6]);

    //Ustal liczbę iteracji próbkowania
    //{ITER}

    int ITER=atoi(argv[7]);

    // Zaalokuj tablicę GPU i tablicę CPU 
    // np. mandel_cpu i mandel_gpu

    size_t mandelSize = POZ * PION * sizeof(int);
    int *mandelCPU = (int*)malloc(mandelSize);
    
    int *mandelGPU;
    cudaMalloc((void**)&mandelGPU, mandelSize);

    time_t start, end;
    // do computations
    
    printf("Computations for rectangle { (%lf %lf), (%lf %lf) }\n",X0,Y0,X1,Y1);
    // miejsce na wstawienie kernela obliczeniowego
    // Najpierw konfiguracja  dla  cudaMandelbrot_1D - mozna poeksperymentowac 
    // z liczba watkow (64,128,256,512)
    // dim3 threadsPerBlock(256,1,1);
    // dim3 numBlocks(PION*POZ/threadsPerBlock.x+1,1,1);
    
    // alternatywna konfiguracja dla cudaMandelbrot_2D - mozna poeksperymentowac 
    int block_width=8;
    int block_height=8;
    dim3 threadsPerBlock(block_width,block_height,1);
    dim3 numBlocks(POZ/block_width+1,PION/block_height+1,1);

    start=clock();

    cudaMandelbrot_2D<<<numBlocks,threadsPerBlock,0>>>(X0,Y0,X1,Y1,POZ,PION,ITER,mandelGPU);
    cudaMemcpy(mandelCPU, mandelGPU, mandelSize, cudaMemcpyDeviceToHost);

    // dopiero po skopiowaniu wyników z GPU na CPU i dopiero potem konczymy pomiar czasu. 
    end=clock();
    printf("Computations took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
    
    start=clock();
    makePicture(mandelCPU, POZ, PION, ITER);
    end=clock();
    printf("Saving took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);

}
    
void computeMandelbrot_2D(float X0, float Y0, float X1, float Y1, int POZ, int PION, int ITER,int *Mandel ){
float    dX=(X1-X0)/(POZ-1);
float    dY=(Y1-Y0)/(PION-1);
float x,y,Zx,Zy,tZx;
int i;

    printf("Computations for rectangle { (%lf %lf), (%lf %lf) }\n",X0,Y0,X1,Y1);

    for (int pion=0; pion<PION; pion++) {
        for (int poz=0;poz<POZ; poz++) {
            x=X0+poz*dX;
            y=Y0+pion*dY;
            Zx=x;
            Zy=y;
            i=0;
            while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) )
            {
                tZx = Zx*Zx-Zy*Zy+x;
                Zy = 2*Zx*Zy+y;
                Zx = tZx;

                i++;
            }
            Mandel[pion*POZ+poz] = i;
        }
    }
}


void computeMandelbrot_1D(float X0, float Y0, float X1, float Y1, int POZ, int PION, int ITER,int *Mandel ){
    float    dX=(X1-X0)/(POZ-1);
    float    dY=(Y1-Y0)/(PION-1);
    float x,y,Zx,Zy,tZx;
    int i;
    int SIZE=POZ*PION;
    int pion, poz;
    
    for (int indx=0;indx<SIZE;indx++) {
        pion=indx / POZ;
        poz=indx % POZ;
        x=X0+poz*dX;
        y=Y0+pion*dY;
        Zx=x;
        Zy=y;
        i=0;
            
        while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) ){
            tZx = Zx*Zx-Zy*Zy+x;
            Zy = 2*Zx*Zy+y;
            Zx = tZx;
            
            i++;
        }
        Mandel[indx] = i;
    }
}


void makePicture(int *Mandel,int width, int height, int MAX){
    
    int red_value, green_value, blue_value;
    
    float scale = 256.0/MAX;
    
    int MyPalette[41][3]={
        {255,255,255}, //0
        {255,255,255}, //1 not used
        {255,255,255}, //2 not used
        {255,255,255}, //3 not used
        {255,255,255}, //4 not used
        {255,180,255}, //5
        {255,180,255}, //6 not used
        {255,180,255}, //7 not used
        {248,128,240}, //8
        {248,128,240}, //9 not used
        {240,64,224}, //10
        {240,64,224}, //11 not used
        {232,32,208}, //12
        {224,16,192}, //13
        {216,8,176}, //14
        {208,4,160}, //15
        {200,2,144}, //16
        {192,1,128}, //17
        {184,0,112}, //18
        {176,0,96}, //19
        {168,0,80}, //20
        {160,0,64}, //21
        {152,0,48}, //22
        {144,0,32}, //23
        {136,0,16}, //24
        {128,0,0}, //25
        {120,16,0}, //26
        {112,32,0}, //27
        {104,48,0}, //28
        {96,64,0}, //29
        {88,80,0}, //30
        {80,96,0}, //31
        {72,112,0}, //32
        {64,128,0}, //33
        {56,144,0}, //34
        {48,160,0}, //35
        {40,176,0}, //36
        {32,192,0}, //37
        {16,224,0}, //38
        {8,240,0}, //39
        {0,0,0} //40
    };
    
    FILE *f = fopen("Mandel.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", width, height);
    for (int j=height-1; j>=0; j--) {
        for (int i=0; i<width; i++) {
            // compute index to the palette
            int indx= (int) floor(5.0f*scale*log2f(1.0f*Mandel[j*width+i]+1));
            red_value=MyPalette[indx][0];
            green_value=MyPalette[indx][2];
            blue_value=MyPalette[indx][1];
            
            fputc(red_value, f);   // 0 .. 255
            fputc(green_value, f); // 0 .. 255
            fputc(blue_value, f);  // 0 .. 255
        }
    }
    fclose(f);
    
}


void makePictureInt(int *Mandel,int width, int height, int MAX){
    
    float scale = 255.0/MAX;
    
    int red_value, green_value, blue_value;

    
    int MyPalette[35][3]={
        {255,0,255},
        {248,0,240},
        {240,0,224},
        {232,0,208},
        {224,0,192},
        {216,0,176},
        {208,0,160},
        {200,0,144},
        {192,0,128},
        {184,0,112},
        {176,0,96},
        {168,0,80},
        {160,0,64},
        {152,0,48},
        {144,0,32},
        {136,0,16},
        {128,0,0},
        {120,16,0},
        {112,32,0},
        {104,48,0},
        {96,64,0},
        {88,80,0},
        {80,96,0},
        {72,112,0},
        {64,128,0},
        {56,144,0},
        {48,160,0},
        {40,176,0},
        {32,192,0},
        {16,224,0},
        {8,240,0},
        {0,0,0}
    };
    
    FILE *f = fopen("Mandel.ppm", "wb");
    
    fprintf(f, "P3\n%i %i 255\n", width, height);
    printf("MAX = %d, scale %lf\n",MAX,scale);
    for (int j=height-1; j>=0; j--) {
        for (int i=0; i<width; i++)
        {
            //if ( ((i%4)==0) && ((j%4)==0) ) printf("%d ",Mandel[j*width+i]);
            //red_value = (int) round(scale*(Mandel[j*width+i])/16);
            //green_value = (int) round(scale*(Mandel[j*width+i])/16);
            //blue_value = (int) round(scale*(Mandel[j*width+i])/16);
            int indx= (int) round(4.0f*scale*log2f(1.0f*Mandel[j*width+i]+1));
            red_value=MyPalette[indx][0];
            green_value=MyPalette[indx][2];
            blue_value=MyPalette[indx][1];
            
            fprintf(f,"%d ",red_value);   // 0 .. 255
            fprintf(f,"%d ",green_value); // 0 .. 255
            fprintf(f,"%d ",blue_value);  // 0 .. 255
        }
        fprintf(f,"\n");
        //if ( (j%4)==0)  printf("\n");

    }
    fclose(f);
    
}


