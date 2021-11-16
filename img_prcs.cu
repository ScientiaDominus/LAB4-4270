#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>

#define PGMHeaderSize           0x40

inline bool loadPPM(const char *file, unsigned char **data, unsigned int *w, unsigned int *h, unsigned int *channels)
{
    FILE *fp = NULL;

    fp = fopen(file, "rb");
         if (!fp) {
              fprintf(stderr, "__LoadPPM() : unable to open file\n" );
                return false;
         }

    // check header
    char header[PGMHeaderSize];

    if (fgets(header, PGMHeaderSize, fp) == NULL)
    {
        fprintf(stderr,"__LoadPPM() : reading PGM header returned NULL\n" );
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
        fprintf(stderr,"__LoadPPM() : File is not a PPM or PGM image\n" );
        *channels = 0;
        return false;
    }

    // parse header, read maxval, width and height
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;

    while (i < 3)
    {
        if (fgets(header, PGMHeaderSize, fp) == NULL)
        {
            fprintf(stderr,"__LoadPPM() : reading PGM header returned NULL\n" );
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

    // check if given handle for the data is initialized
    if (NULL != *data)
    {
        if (*w != width || *h != height)
        {
            fprintf(stderr, "__LoadPPM() : Invalid image dimensions.\n" );
        }
    }
    else
    {
        *data = (unsigned char *) malloc(sizeof(unsigned char) * width * height * *channels);
        if (!data) {
         fprintf(stderr, "Unable to allocate hostmemory\n");
         return false;
        }
        *w = width;
        *h = height;
    }

    // read and close file
    if (fread(*data, sizeof(unsigned char), width * height * *channels, fp) == 0)
    {
        fprintf(stderr, "__LoadPPM() : read data returned error.\n" );
        fclose(fp);
        return false;
    }

    fclose(fp);

    return true;
}

inline bool savePPM(const char *file, unsigned char *data, unsigned int w, unsigned int h, unsigned int channels)
{
    assert(NULL != data);
    assert(w > 0);
    assert(h > 0);

    std::fstream fh(file, std::fstream::out | std::fstream::binary);

    if (fh.bad())
    {
        fprintf(stderr, "__savePPM() : Opening file failed.\n" );
        return false;
    }

    if (channels == 1)
    {
        fh << "P5\n";
    }
    else if (channels == 3)
    {
        fh << "P6\n";
    }
    else
    {
        fprintf(stderr, "__savePPM() : Invalid number of channels.\n" );
        return false;
    }

    fh << w << "\n" << h << "\n" << 0xff << std::endl;

    for (unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i)
    {
        fh << data[i];
    }

    fh.flush();

    if (fh.bad())
    {
        fprintf(stderr,"__savePPM() : Writing data failed.\n" );
        return false;
    }

    fh.close();

    return true;
}

#define TILE_W      16
#define TILE_H      16
#define Rx          2                       // filter radius in x direction
#define Ry          2                       // filter radius in y direction
#define FILTER_W    (Rx*2+1)                // filter diameter in x direction
#define FILTER_H    (Ry*2+1)                // filter diameter in y direction
#define S           (FILTER_W*FILTER_H)     // filter size
#define BLUR_SIZE   1
#define BLOCK_W     (TILE_W+(2*Rx))
#define BLOCK_H     (TILE_H+(2*Ry))

__global__ void box_filter(const unsigned char *in, unsigned char *out, const unsigned int w, const unsigned int h){
    //Indexes
    const int x = blockIdx.x * TILE_W + threadIdx.x - Rx;       // x image index
    const int y = blockIdx.y * TILE_H + threadIdx.y - Ry;       // y image index
    const int d = y * w + x;                                    // data index

    //shared mem
    __shared__ float shMem[BLOCK_W][BLOCK_H];
    if(x<0 || y<0 || x>=w || y>=h) {            // Threads which are not in the picture just write 0 to the shared mem
        shMem[threadIdx.x][threadIdx.y] = 0;
        return; 
    }
    shMem[threadIdx.x][threadIdx.y] = in[d];
    __syncthreads();

    // box filter (only for threads inside the tile)
    if ((threadIdx.x >= Rx) && (threadIdx.x < (BLOCK_W-Rx)) && (threadIdx.y >= Ry) && (threadIdx.y < (BLOCK_H-Ry))) {
        float sum = 0;
        for(int dx=-Rx; dx<=Rx; dx++) {
            for(int dy=-Ry; dy<=Ry; dy++) {
                sum += shMem[threadIdx.x+dx][threadIdx.y+dy];
            }
        }
    out[d] = sum / S;       
    }
}
__global__ void Gaussian(unsigned char *in, unsigned char *out, int w, int h) 
{ 
    int Col = blockIdx.x * blockDim.x + threadIdx.x; 
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) 
    { 
        int pixVal = 0; 
        int pixels = 0;
        // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box 
            for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) 
                { 
                    for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) 
                    {
                        int curRow = Row + blurRow; 
                        int curCol = Col + blurCol; 

                        // Verify we have a valid image pixel 
                        if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) 
                            { 
                                pixVal += in[curRow * w + curCol]; 
                                pixels++; // Keep track of number of pixels in the avg 
                            } 
                    } 
                } 
            // Write our new pixel value out 
        out[Row * w + Col] = (unsigned char)(pixVal / pixels); 
    } 
}

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(){
    unsigned char *data=NULL, *d_idata=NULL, *d_odata=NULL;
    unsigned int w,h,channels;

    if(! loadPPM("puppy.pgm", &data, &w, &h, &channels)){
        fprintf(stderr, "Failed to open File\n");
        exit(EXIT_FAILURE);
    }

    printf("Loaded file with   w:%d   h:%d   channels:%d \n",w,h,channels);

    unsigned int numElements = w*h*channels;
    size_t datasize = numElements * sizeof(unsigned char);

    // Allocate the Device Memory
    printf("Allocate Devicememory for data\n");
    checkCudaErrors(cudaMalloc((void **)&d_idata, datasize));
    checkCudaErrors(cudaMalloc((void **)&d_odata, datasize));

    // Copy to device
    printf("Copy idata from the host memory to the CUDA device\n");
    checkCudaErrors(cudaMemcpy(d_idata, data, datasize, cudaMemcpyHostToDevice));

    // Launch Kernel
    int GRID_W = w/TILE_W +1;
    int GRID_H = h/TILE_H +1;
    dim3 threadsPerBlock(BLOCK_W, BLOCK_H);
    dim3 blocksPerGrid(GRID_W,GRID_H);
    printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    box_filter<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata, w,h);
    checkCudaErrors(cudaGetLastError());

    // Copy data from device to host
    printf("Copy odata from the CUDA device to the host memory\n");
    checkCudaErrors(cudaMemcpy(data, d_odata, datasize, cudaMemcpyDeviceToHost));
    printf("Free Device memory\n");
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    printf("Save Picture\n");
    bool saved = false;
    if      (channels==1)
    {
        saved = savePPM("output_box.pgm", data, w,  h,  channels);
        //saved = savePPM("output.pgm", data2, w,  h,  channels);
    }
    else if (channels==3)
    {
        saved = savePPM("output_box.ppm", data, w,  h,  channels);
        //saved = savePPM("output.ppm", data2, w,  h,  channels);
    }
    else fprintf(stderr, "ERROR: Unable to save file - wrong channel!\n");

    checkCudaErrors(cudaMalloc((void **)&d_idata, datasize));
    checkCudaErrors(cudaMalloc((void **)&d_odata, datasize));
    printf("Copy idata from the host memory to the CUDA device\n");
    checkCudaErrors(cudaMemcpy(d_idata, data, datasize, cudaMemcpyHostToDevice));
    printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    Gaussian<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata, w, h);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(data, d_odata, datasize, cudaMemcpyDeviceToHost));
    // Free Device memory
    printf("Free Device memory\n");
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));


    // Save Picture
    printf("Save Picture\n");
    saved = false;
    if      (channels==1)
    {
        saved = savePPM("output_gauss.pgm", data, w,  h,  channels);
        //saved = savePPM("output.pgm", data2, w,  h,  channels);
    }
    else if (channels==3)
    {
        saved = savePPM("output_gauss.ppm", data, w,  h,  channels);
        //saved = savePPM("output.ppm", data2, w,  h,  channels);
    }
    else fprintf(stderr, "ERROR: Unable to save file - wrong channel!\n");

    // Free Host memory
    printf("Free Host memory\n");
    free(data);

    if (!saved){
        fprintf(stderr, "Failed to save File\n");
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
}