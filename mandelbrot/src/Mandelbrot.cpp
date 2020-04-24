//
//  Mandelbrot.cpp
//  
//
//  Created by Witold Rudnicki
//
//

#include "Mandelbrot.h"
#include <math.h>


int main(int argc, char **argv) {

    if (argc!=8) {
        printf("Wywołanie %s LD_Re, LD_Im, PG_Re, PG_Im, Poziom, Pion, Iteracje\n ",argv[0]);
    }
    //Ustaw obszar obliczeń
    //{X0,Y0} - lewy dolny róg
    double X0=atof(argv[1]);
    double Y0=atof(argv[2]);

    //{X1,Y1} - prawy górny róg
    double X1=atof(argv[3]);
    double Y1=atof(argv[4]);

    //Ustal rozmiar w pikselach
    //{POZ,PION}
    int POZ=atoi(argv[5]);
    int PION=atoi(argv[6]);

    //Ustal liczbę iteracji próbkowania
    //{ITER}

    int ITER=atoi(argv[7]);


    //Zaalokuj tablicę do przechowywania wyniku

    int *Iters;
    Iters= (int*) malloc(sizeof(int)*POZ*PION);
    
    time_t start, end;
    // do computations
    
    printf("Computations for rectangle { (%lf %lf), (%lf %lf) }\n",X0,Y0,X1,Y1);
    start=clock();
    int SUM = computeMandelbrot(X0,Y0,X1,Y1,POZ,PION,ITER,Iters);
    end=clock();
    
    printf("\nTotal %d iterations took %lf s\n\n",SUM,1.0*(end-start)/CLOCKS_PER_SEC);
    //makePictureInt(Iters, POZ, PION, ITER);
    
    start=clock();
    //makePictureInt(Iters, POZ, PION, ITER);
    makePicture(Iters, POZ, PION, ITER);
    end=clock();
    printf("Saving took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);

}
    
int computeMandelbrot(double X0, double Y0, double X1, double Y1, int POZ, int PION, int ITER,int *Mandel ){
double    dX=(X1-X0)/(POZ-1);
double    dY=(Y1-Y0)/(PION-1);
double x,y,Zx,Zy,tZx;
int SUM=0;
int i;

    printf("Computations for rectangle { (%lf %lf), (%lf %lf) }\n",X0,Y0,X1,Y1);

    for (int pion=0; pion<PION; pion++) {
        for (int poz=0;poz<POZ; poz++) {
            x=X0+poz*dX;
            y=Y0+pion*dY;
            Zx=x;
            Zy=y;
            i=0;
            //printf("%d %d %lf %lf\n",pion,poz,y,x);
            while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) )
            {
                tZx = Zx*Zx-Zy*Zy+x;
                Zy = 2*Zx*Zy+y;
                Zx = tZx;

                i++;
            }
            Mandel[pion*POZ+poz] = i;
            SUM+=i;
        }
        //printf("Line %d, sum=%d\n",pion,SUM);
    }
    return SUM;
}


int computeMandelbrot2(double X0, double Y0, double X1, double Y1, int POZ, int PION, int ITER,int *Mandel ){
    double    dX=(X1-X0)/(POZ-1);
    double    dY=(Y1-Y0)/(PION-1);
    double x,y,Zx,Zy,tZx;
    int SUM=0;
    int i;
    int SIZE=POZ*PION;
    int pion, poz;
    
    for (int indx=0;indx<SIZE;indx++) {
        pion=indx / POZ;
        poz=indx % POZ;
        //printf("%d %d %d \n",indx,pion,poz);
        x=X0+poz*dX;
        y=Y0+pion*dY;
        //printf("%lf %lf\n",x,y);
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
        SUM+=i;
    }
    return SUM;
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
            int indx= (int) floor(5.0*scale*log2f(1.0f*Mandel[j*width+i]+1));
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
            int indx= (int) round(4*scale*log2(Mandel[j*width+i]+1));
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
