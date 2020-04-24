//
//  scalar_product.c
//  
//
//  Created by Witold Rudnicki on 09.03.2020.
//

#include "scalar_product.h"
int ReadCSV(char* CSVfile,float** CNV) {
// We know the size of the data, we define sufficiently large buffers for this case.
//
    FILE* CNVfile;
    float* row;
    const int buf_size=2000000;
    const int max_cols=100000;
    char buffer[buf_size];
    char* tmp;
    //char head[110]; //for debugging only
    int col_count;
    
    row = (float*) malloc(max_cols*sizeof(float));
    printf("Reading %s\n",CSVfile);
    CNVfile = fopen(CSVfile,"r");
    int line_count =0;
    
    int row_count;
    while ((tmp=fgets(buffer, 1999999, CNVfile))!=NULL){
        line_count++;
        if (line_count>1){
            //we skip header line, hence we start from line_count = 2 here.
            row_count = line_count-2;
            //for (int i=0;i<30;i++) head[i]=buffer[i];
            //printf("line %d starts as %30s\n",line_count,head);
            col_count=-1;
            char *col = strtok(buffer, ",");
            while (col) {
                if (col_count >= 0) {
                    row[col_count]=atof(col);
                }
                col = strtok(NULL, ",");
                col_count++;
            }
            //printf("converted entire line of %d columns\n",col_count);
            //printf("%8.3f, %8.3f %8.3f\n",row[0],row[1],row[2]);
            CNV[row_count]= (float*) malloc((col_count+1)*sizeof(float));
            for (int i=0;i<=col_count;i++) CNV[row_count][i]=row[i];
            //printf("row_count %d\n",row_count);
            //printf("&CNV[%d] = %d\n",row_count,CNV[row_count]);
            //for (int i=0;i<10;i++) printf("%f, ",CNV[row_count][i]);
            //printf("\n");
        }
    }
    //printf("%i\n",CNV);
    //printf("%f\n",CNV[0][0]);
    fclose(CNVfile);
    //for (int i=0;i<10;i++) {
    //    for (int j=0;j<10;j++) printf("%f, ",CNV[i][j]);
    //    printf("\n");
    //}
    return(col_count);
}


float scalar(float *x, float* y,int len){
    //printf("Multiplying vectors\n");
    float sum=0.0f;
    for (int i=0;i<len;i++) sum+= x[i]*y[i];
    return(sum);
}

void scalar_gpu(float *x, float* y, int len, float* z, float* sum){
    //printf("Multiplying vectors\n");
    //float* z;
    //z = malloc(sizeof(float)*len);
    *sum =0.0f;
    
    for (int i=0;i<len;i++) z[i] = x[i]*y[i];

    for (int i=0;i<len;i++) *sum += z[i];
    // tu robimy redukcję na GPU
    //return(sum);
}


double scalar_d(float *x, float* y,int len){
    //printf("Multiplying vectors\n");
    double sum=0.0;
    for (int i=0;i<len;i++) sum+= x[i]*y[i];
    return(sum);
}


int main(int argc, char** argv){
    const int ROWS=145;
    //float x[1], y[1];
    float* CNV[ROWS];
    float* normCNV[ROWS];
    //float scl;
    //double scl_d;
    float* SimMat;
    
    int len;
    if (argc==2) {
        len=ReadCSV(argv[1],CNV);
        // ReadCSV returns the length of the vectors,
        // It also fills two-dimensional array CNV.
        // The rows of CNV are allocated in the function.
        //
        // Allocation of the memory for the normalized version of CNV array.
        for (int i=0;i<ROWS;i++) normCNV[i]= (float*) malloc(len*sizeof(float));
        // Normalization
        normalize(CNV,normCNV,ROWS,len);
        //printf("Lengths of normalized vectors:\n");
        //for (int i=0;i<ROWS;i++){
        //    scl = scalar(normCNV[i],normCNV[i],len);
        //    printf("%12.8f ",scl);
        // }
        //printf("\n");
        //normalize_d(CNV,normCNV,ROWS,len);
        //for (int i=0;i<ROWS;i++){
        //    scl_d = scalar_d(normCNV[i],normCNV[i],len);
        //    printf("%12.8lf ",scl_d);
        //}
        //
        // Computation of similarity matrix
        //
        SimMat= (float*) malloc(ROWS*ROWS*sizeof(float));
        //similarity(normCNV,SimMat,ROWS,len);
        similarity_gpu(normCNV,SimMat,ROWS,len);
        return(0);
    }
    else {
        printf("Wrong number of arguments\n");
        printf("Usage: %s filename\nExiting\n",argv[0]);
        exit(1);
    }
}

void normalize(float** sourceMat,float**  destMat, int Size, int Len) {
    int i,j;
    float Sum;
    // Zdefiniuj timery
    time_t start, end;
    // do computations
    printf("function normalize():\n");
    start=clock();
    for (i=0;i<Size;i++){
        Sum=scalar(sourceMat[i],sourceMat[i],Len);
        Sum=sqrt(Sum);
        for (j=0;j<Len;j++) destMat[i][j]=sourceMat[i][j]/Sum;
        //printf("%8.4f ",Sum);
    }
    end=clock();
    printf("computations took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
    
}

void normalize_d(float** sourceMat,float**  destMat, int Size, int Len) {
    int i,j;
    double Sum;
    float Sum_f;
    // Zdefiniuj timery
    time_t start, end;
    // do computations
    start=clock();
    for (i=0;i<Size;i++){
        Sum=scalar_d(sourceMat[i],sourceMat[i],Len);
        Sum_f= (float) sqrt(Sum);
        for (j=0;j<Len;j++) destMat[i][j]=sourceMat[i][j]/Sum_f;
        printf("%8.4lf ",Sum);
    }
    end=clock();
    printf("\nComputations took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
}


void similarity(float** sourceMat, float* simMat, int Size, int Len){
    int i,j;
    float Sum, Min, Max;
    Sum=Max=0.0f;
    Min=1.0f;
    time_t start, end;
    // do computations
    printf("function similarity()\n:");
    start=clock();
    for (i=0;i<Size;i++) {
        for (j=0;j<Size;j++) {
            simMat[i*Size+j]=scalar(sourceMat[i],sourceMat[j],Len);
            //Sum+=simMat[i*Size+j];
            //if (Min>fabs(simMat[i*Size+j])) Min=fabs(simMat[i*Size+j]);
            //if ( (i!=j) & (Max<simMat[i*Size+j]) ) Max=simMat[i*Size+j];
        }
    }
    end=clock();
    printf("\ncomputations took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
    printf("Minimum similarity(%%): %f\n",Min*100);
    printf("Maximum similarity(%%): %f\n",Max*100);
    printf("Average similarity(%%): %f\n",(Sum-Size)/(Size*(Size-1))*100);
}

void similarity_gpu(float** sourceMat, float* simMat, int Size, int Len){
    int i,j, indx;
    float Sum, Min, Max;
    Sum=Max=0.0f;
    Min=1.0f;
    float *gpuMat;
    float *cpuMat;
    
    float *simMat_cpu;
    float *simMat_gpu;
    cudaError_t status;
    time_t start, end;
    
    start=clock();
    // Make a local copy of the CPU matrix
    // It will be easier to copy it in a single operation to GPU
    status = cudaMallocHost((void**) &cpuMat, sizeof(float) * Size*Len);
    if (status != cudaSuccess) { cout << cudaGetErrorString(status) << endl; };
    // copy vectors from sourceMat to cpuMat
    indx=0;
    for (i =0;i<Size;i++) {
        for (j=0;j<Len;j++) {
            cpuMat[indx]=sourceMat[i][j];
            indx++;
        }
    }
    end=clock();
    printf("\n Memory allocation and copying  %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
    // Allocat a local CPU similarity matrix
    status = cudaMallocHost((void**) &simMat_cpu, sizeof(float) * Size*Size);
    if (status != cudaSuccess) { cout << cudaGetErrorString(status) << endl; };

    // Allocate GPU tables
    status = cudaMalloc((void**) &gpuMat, sizeof(float) * Size*Len);
    if (status != cudaSuccess) { cout << cudaGetErrorString(status) << endl; };

    status = cudaMalloc((void**) &simMat_gpu, sizeof(float) * Size*Size);
    if (status != cudaSuccess) { cout << cudaGetErrorString(status) << endl; };

    // Copy CPU Table to GPU
    status = cudaMemcpy(gpuMat, cpuMat, sizeof(float) * Size*Len, cudaMemcpyHostToDevice);
    if (status != cudaSuccess){ cout << cudaGetErrorString(status) << endl; };
    
    
    // do computations
    printf("function similarity()\n:");
    start=clock();
    for (i=0;i<Size;i++) {
        for (j=0;j<Size;j++) {
            simMat[i*Size+j]=scalar(sourceMat[i],sourceMat[j],Len);
        }
    }
    end=clock();
    printf("\n CPU computations took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
    
    Min =1.0;
    Max =0.0;
    float r, r2;
    Sum = 0;

    for (i=0;i<Size;i++) {
        for (j=i+1;j<Size;j++) {
            r = simMat[i*Size+j];
            r2 = r*r;
            if (r2 < Min) Min=r2;
            if (r2>Max) Max = r2;
            Sum += r2;
        }
    }
    
    printf("Minimum similarity(%%): %f\n",Min*100);
    printf("Maximum similarity(%%): %f\n",Max*100);
    printf("Average similarity(%%): %f\n",Sum/(Size*(Size-1))/2*100);
    #define KERNEL 1
    if (KERNEL == 1) {
        // kernel 1
        //int limit =29;
        int limit =Size;
        start=clock();
        for (i=0;i<limit;i++) {
            for (j=0;j<limit;j++) {
                scalar_1<<< 1, 256, 0>>>(gpuMat , i, j , Len, Size, simMat_gpu );
                //scalar_2<<< 1, BlockSize, 0>>>(gpuMat , i, j , Len, Size, simMat_gpu );
                //simMat[i*Size+j]=scalar(sourceMat[i],sourceMat[j],Len);            
            }
        }
        status = cudaMemcpy(simMat_cpu, simMat_gpu, sizeof(int)* Size*Size, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess){ cout << cudaGetErrorString(status) << endl; };
        end=clock();
        printf("\n GPU computations with kernel 1 for %d vectors took %lf s\n\n",limit,1.0*(end-start)/CLOCKS_PER_SEC);
    }
    if (KERNEL == 2) {
        // kernel 2
        start=clock();
        for (i=0;i<Size;i++) {
            for (j=0;j<Size;j++) {
                scalar_2<<< 1, BlockSize, 0>>>(gpuMat , i, j , Len, Size, simMat_gpu );           
            }
        }
        status = cudaMemcpy(simMat_cpu, simMat_gpu, sizeof(int)* Size*Size, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess){ cout << cudaGetErrorString(status) << endl; };
        end=clock();
        printf("\n GPU computations with kernel 2 took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
        float r = correlation(simMat,simMat_cpu,(Size*Size));
        printf("Correlation between CPU and GPU %f\n",r);
    }

    Min =1.0;
    Max =0.0;
    Sum = 0;

    for (i=0;i<Size;i++) {
        for (j=i+1;j<Size;j++) {
            r = simMat_cpu[i*Size+j];
            r2 = r*r;
            if (r2 < Min) Min=r2;
            if (r2>Max) Max = r2;
            Sum += r2;
        }
    }
    
    printf("Minimum similarity(%%): %f\n",Min*100);
    printf("Maximum similarity(%%): %f\n",Max*100);
    printf("Average similarity(%%): %f\n",Sum/(Size*(Size-1))/2*100);

    for (i=0;i<20;i++)   printf("%d : %8.5f ",i*Size+i,simMat[i*Size+i]);       
    printf("\n");
    for (i=0;i<20;i++)   printf("%d : %8.5f ",i*Size+i,simMat_cpu[i*Size+i]);       
    printf("\n");
    

}


__global__ void scalar_1(float *Mat , int ind_x, int ind_y , int Len, int Size, float *resMat ){
    size_t s = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i;
    int ad_x;
    int ad_y;
    float res;
    // Wektory x i y są fragmentami ciągłego obszaru pamięci w tablicy Mat.
    // Musimy ręcznie policzyć sobie ich adresy i odpowiednie wartości czytać z tablicy Mat.
    
    
    res = 0.0f;
    if (s==0){
        //    *out = 0;
        ad_x = ind_x*Len;
        ad_y = ind_y*Len;
        //printf("i = %d  j = %d ad_x = %d ad_y = %d\n",ind_x,ind_y,ad_x,ad_y);
        //for (i=0; i<Len; i++) {
        for (i=0; i<Len; i++) {
            res+= Mat[ad_x]*Mat[ad_y];
            ad_x++;
            ad_y++;
        }
        // wpisujemy odpowiedni wynik do tablicy wynikowej
        resMat[ind_x*Size+ind_y]=res;
    }
    
}

__global__ void scalar_2(float *Mat, int ind_x, int ind_y, int Len, int Size, float *resMat ){
    size_t s = threadIdx.x + blockIdx.x * blockDim.x;
    int sID = threadIdx.x;
    int i;
    int stride;
    float sum=0;
    int indx;
    int ad_x=ind_x*Len;
    int ad_y=ind_y*Len;
    int loc_x;
    int loc_y;
    
    stride = Len / blockDim.x +1;
    
    
    // We use a single block to process single pair of vectors
    
    __shared__ float pom[BlockSize];
    
    pom[sID] = 0;
    // Najpierw każdy wątek czyta 'stride' wartości z wektorów x i y
    // i wykonuje fragment obliczeń do iloczynu skalarnego
    // Potem wszystkie wątki robią redukcję.
    // Pamiętamy, że każdy wektor jest w rzeczywistości fragmentem pamięci w jednowymiarowej
    // tablicy Mat. Dlatego musimy policzyć odpowiedni adres każdego elementu z tej tablicy
    for (i=0;i<stride;i++) {
        indx = s*stride+i; // Wyliczamy dla każdego wątku numer elementu w wektorach x i y
        loc_x = ad_x + indx; // Wyliczamy numery elementów w tablicy Mat odpowiadające
        loc_y = ad_y + indx; // elementom o numerze indx w wektorach x i y
        if (indx<Len) {
            sum+=Mat[loc_x]*Mat[loc_y];
        }
    }
    pom[sID] = sum;
    __syncthreads(); // synchronizujemy wątki - wszystkie wątki muszą skończyć swoje wczytywanie
    // zanim przejdziemy dalej
    
    // redukcja numer 2
    for (i=1; i<blockDim.x; i*=2){
        if (sID%(2*i)==0){
            pom[sID] += pom[sID + i];
        }
        __syncthreads();
    }
    if (sID==0) {
        // wpisujemy odpowiedni wynik do tablicy wynikowej
        resMat[ind_x*Size+ind_y]=pom[0];
    }

}






#define blockSize 512
#define real float


__global__ void redukcja_1(int N, real* v)
{
 size_t s = threadIdx.x + blockIdx.x * blockDim.x;
 size_t i;

 real p = 0;
 if (s==0){
//	*out = 0;
	for (i=0; i<N; i++)
        //p+= x[i]*y[i];
		p += v[i];
 	v[0] = p;		
 }		
}

__global__ void redukcja_2(int N, real* v, real* out)
{
 size_t s = threadIdx.x + blockIdx.x * blockDim.x;
 int sID = threadIdx.x;
 size_t i;

 __shared__ real pom[blockSize];
 
 pom[sID] = 0;
 if (s<N)
	 pom[sID] = v[s];
 __syncthreads();

 for (i=1; i<blockSize; i*=2){
 	if (sID%(2*i)==0){
		pom[sID] += pom[sID + i];
 	}		
 	__syncthreads();		
 }
 if (sID==0) out[blockIdx.x] = pom[0];
}

float correlation(float *x,float *y,int Size){
    float Sumx, Sumy, Sumxy, Sumxx, Sumyy; 
    float mx, my; 
    float tx, ty;
    float r;

    Sumx =0;
    Sumy =0;
    Sumxy =0;
    Sumxx=0;
    Sumyy=0;

    for (int i=0; i<Size;i++) {
        tx = x[i];
        ty = y[i];
        Sumx += tx;
        Sumy += ty;
        Sumxy += tx*ty;
        Sumxx += tx*tx;
        Sumyy += ty*ty;
    }
    mx = Sumx/Size;
    my = Sumy/Size;
    r = (Sumxx - Size*mx*my)/(sqrt( (Sumxx -Size*mx*mx)*(Sumyy-Size*my*my) ));
    return(r);


}
