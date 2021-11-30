#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>


#define H_Size           0x40
#define TILE_WIDTH      16
#define TILE_HEIGHT      16
#define x_radius          3                       
#define y_radius          3  
#define BLUR_SIZE   6                     
#define FILTER_WIDTH    (x_radius*2+1)                
#define FILTER_HEIGHT    (y_radius*2+1)                
#define S           (FILTER_WIDTH*FILTER_HEIGHT)    
#define BLOCK_W     (TILE_WIDTH+(2*x_radius))
#define BLOCK_H     (TILE_HEIGHT+(2*y_radius))

/*
IMPORTANT INFORMATION:
This program will ONLY read pgm format images to blur. 

In order to run the program you must use the command line arguments to enter the name of the file you wish to process.

This program can only process one image at a time. 

The program applies the box_filter and the Gaussian Blur filter to the image you entered. The result of these filters will be saved in "output_box.pgm" and "output_gauss.pgm."

cuda8 environment was used for this code.   
*/



bool LoadImage(const char *file, unsigned char **data, unsigned int *w, unsigned int *h, unsigned int *channels)
{
    FILE *fp = NULL;
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;
    char header[H_Size];

    fp = fopen(file, "rb");
         if (!fp) {
             printf("ERROR READING FILE FAILED\n");
                return false;
         }


    if (fgets(header, H_Size, fp) == NULL)
    {
        return false;
    }

    if (strncmp(header, "P5", 2) == 0)
    {
        *channels = 1;
    }
    else if (strncmp(header, "P6", 2) == 0)
    {
        *channels = 3;
    }
    else
    {
        *channels = 0;
        return false;
    }

    while (i < 3)
    {
        if (fgets(header, H_Size, fp) == NULL)
        {
            return false;
        }

        if (header[0] == '#')
        {
            continue;
        }

        if (i == 0)
        {
            i += sscanf(header, "%u %u %u", &width, &height, &maxval);
        }
        else if (i == 1)
        {
            i += sscanf(header, "%u %u", &height, &maxval);
        }
        else if (i == 2)
        {
            i += sscanf(header, "%u", &maxval);
        }
    }

    if (NULL != *data)
    {
        if (*w != width || *h != height)
        {
            return false;
        }
    }
    else
    {
        *data = (unsigned char *) malloc(sizeof(unsigned char) * width * height * *channels);
        if (!data) {
         return false;
        }
        *w = width;
        *h = height;
    }

    // read and close file
    if (fread(*data, sizeof(unsigned char), width * height * *channels, fp) == 0)
    {
        fclose(fp);
        return false;
    }

    fclose(fp);

    return true;
}

bool SaveFile(const char *file, unsigned char *data, unsigned int w, unsigned int h, unsigned int channels)
{
    std::fstream Image(file, std::fstream::out | std::fstream::binary);

    if (Image.bad())
    {
        return false;
    }

    if (channels == 1)
    {
        Image << "P5\n";
    }
    else if (channels == 3)
    {
        Image << "P6\n";
    }
    else
    {
        Image.close();
        return false;
    }

    Image << w << "\n" << h << "\n" << 0xff << std::endl;

    for (unsigned int i = 0; (i < (w*h*channels)) && Image.good(); ++i)
    {
        Image << data[i];
    }
    if (Image.bad())
    {
        Image.close();
        return false;
    }
    Image.close();

    return true;
}

//This function implements the box_filter algorithm
__global__ void box_filter(const unsigned char *in, unsigned char *out, const unsigned int w, const unsigned int h){
    const int x = blockIdx.x * TILE_WIDTH + threadIdx.x - x_radius;       
    const int y = blockIdx.y * TILE_HEIGHT + threadIdx.y - y_radius;       
    const int d = y * w + x;                                    

    __shared__ float Memory[BLOCK_W][BLOCK_H];
    if(x<0 || y<0 || x>=w || y>=h) {            
        Memory[threadIdx.x][threadIdx.y] = 0;
        return; 
    }
    Memory[threadIdx.x][threadIdx.y] = in[d];
    __syncthreads();

    if ((threadIdx.x >= x_radius) && (threadIdx.x < (BLOCK_W-x_radius)) && (threadIdx.y >= y_radius) && (threadIdx.y < (BLOCK_H-y_radius))) {
        float sum = 0;
        for(int dx=-x_radius; dx<=x_radius; dx++) {
            for(int dy=-y_radius; dy<=y_radius; dy++) {
                sum += Memory[threadIdx.x+dx][threadIdx.y+dy];
            }
        }
    out[d] = sum / S;       
    }
}
//This function implements the Gaussian blur algorithm on the GPU
__global__ void Gaussian(unsigned char *in, unsigned char *out, const unsigned int w, const unsigned int h) 
{ 
    int Col = blockIdx.x * blockDim.x + threadIdx.x; 
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) 
    { 
        int value = 0; 
        int pixelCount = 0;
            for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) 
                { 
                    for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) 
                    {
                        int curRow = Row + blurRow; 
                        int curCol = Col + blurCol; 
 
                        if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) 
                            { 
                                value += in[curRow * w + curCol]; 
                                pixelCount++; 
                            } 
                    } 
                } 
        out[Row * w + Col] = (unsigned char)(value / pixelCount); 
    } 
}

int main(int argc, char* argv[]){
    unsigned char *data=NULL, *d_in=NULL, *d_out=NULL;
    unsigned int w,h,channels;
    int GRID_W = 0;
    int GRID_H = 0;

    if(argc < 2)
    {
        int i =0;
        while(i != 10)
        {
            printf("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING\n");
            i++;
        }
        printf("ERROR: NO INPUT FILE SELECTED! USE %s <INPUT FILE> \n\n", argv[0]);
        exit(1);
    }
    char* filename = (char*)malloc(strlen(argv[1])*sizeof(char));
    strcpy(filename, argv[1]);

    if(! LoadImage(filename, &data, &w, &h, &channels)){
        exit(EXIT_FAILURE);
    }

    GRID_W = w/TILE_WIDTH +1;
    GRID_H = h/TILE_HEIGHT +1;

    printf("File loaded, Starting process\n");

    unsigned int numElements = w*h*channels;
    size_t memSize = numElements * sizeof(unsigned char);

    cudaMalloc((void **)&d_in, memSize);
    cudaMalloc((void **)&d_out, memSize);

    cudaMemcpy(d_in, data, memSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_W, BLOCK_H);
    dim3 blocksPerGrid(GRID_W,GRID_H);
    box_filter<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, w,h);
    cudaMemcpy(data, d_out, memSize, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("Box Filter complete, saving image\n");
    bool saved = false;
    if (channels==1)
    {
        saved = SaveFile("output_box.pgm", data, w,  h,  channels);
    }
    else if (channels==3)
    {
        saved = SaveFile("output_box.ppm", data, w,  h,  channels);
    }
    else fprintf(stderr, "ERROR: Unable to save file - wrong channel!\n");

    cudaMalloc((void **)&d_in, memSize);
    cudaMalloc((void **)&d_out, memSize);
    cudaMemcpy(d_in, data, memSize, cudaMemcpyHostToDevice);
    Gaussian<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, w, h);
    cudaMemcpy(data, d_out, memSize, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);


    // Save Picture
    printf("Gaussian Blur complete, saving image\n");
    saved = false;
    if (channels==1)
    {
        saved = SaveFile("output_gauss.pgm", data, w,  h,  channels);
    }
    else if (channels==3)
    {
        saved = SaveFile("output_gauss.ppm", data, w,  h,  channels);
    }
    else fprintf(stderr, "ERROR: Unable to save file - unrecognized format\n");

    free(data);

    if (!saved){
        fprintf(stderr, "Failed to save File\n");
        exit(EXIT_FAILURE);
    }
    free(filename);

    printf("Done\n");
}