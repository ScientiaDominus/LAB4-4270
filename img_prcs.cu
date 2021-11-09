#include <stdio.h>
#include <stdlib.h>

__global__ void imageProcess()
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    
}

int main(int argc, char* argv[])
{

}