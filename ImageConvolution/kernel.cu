#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <vector>

#include "lodepng.h"
#include "CudaTiming.cuh"

#include "cuda_profiler_api.h"
#include <cuda_fp16.h>
#include <random>

using std::vector;

#define CONV_SIDE_LENGTH 49

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4


__constant__ float constantConv[CONV_SIDE_LENGTH * CONV_SIDE_LENGTH];

__device__ unsigned char clamp(float sum)
{
    //return (sum >= 255) * 255 + (sum > 0 && sum < 255) * static_cast<unsigned char>(sum);
    if (sum >= 255)
    {
        return 255;
    }
    else if (sum > 0)
    {
        return static_cast<unsigned char>(sum);
    }
    else {
        return 0;
    }
}
__global__ void Convolve(unsigned char* out, unsigned char* pixels, int width, int height)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int color = threadIdx.z;

    int col_offset = CONV_SIDE_LENGTH / 2;
    int row_offset = CONV_SIDE_LENGTH / 2;

    
        float sum = 0.0;

        for (int i = row - row_offset; i <= row + row_offset; i++)
        {
            if (i < 0 || i >= height)
                continue;
            for (int j = col - col_offset; j <= col + col_offset; j++)
            {
                if (j < 0 || j >= width)
                    continue;
                int convRow = i - (row - row_offset);
                int convCol = j - (col - col_offset);
                int convIdx = convRow * CONV_SIDE_LENGTH + convCol;


                int pixelIdx = (i * width + j) * 4 + color;

                unsigned char pixelVal = pixels[pixelIdx];
                sum = sum + (constantConv[convIdx] * pixelVal);
                

            }
        }
        int idx = (row * width + col) * 4 + color;
       

        out[idx] = clamp(sum);
    

}

__global__ void ConvolveSharedMemory(unsigned char* out, const unsigned char* pixels, const int width, const int height)
{

    int row = blockIdx.y * BLOCK_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_X + threadIdx.x;
    int color = threadIdx.z;

    int idx = (row * width + col) * 4 + color;
    if (idx >= width * height * BLOCK_Z) return;

    int convOffset = CONV_SIDE_LENGTH / 2;

    const int shared_block_width = BLOCK_X + CONV_SIDE_LENGTH - 1;
    const int shared_block_height = BLOCK_Y + CONV_SIDE_LENGTH - 1;
    const int shared_block_size = shared_block_width * shared_block_height;

    __shared__ unsigned char shared_block[shared_block_height * shared_block_width * BLOCK_Z];

    //set shared memory
    int sub_pixel_idx = threadIdx.y * blockDim.x + threadIdx.x;

    while (sub_pixel_idx < shared_block_size)
    {
        int x = sub_pixel_idx % shared_block_width;
        int y = sub_pixel_idx / shared_block_width;

        int x_global = (x - convOffset) + blockIdx.x * BLOCK_X;
        int y_global = (y - convOffset) + blockIdx.y * BLOCK_Y;

        if (x_global >= 0 && y_global >= 0 && x_global < width && y_global < height)
        {
            shared_block[(y * shared_block_width + x) * BLOCK_Z + color] = pixels[(y_global * width + x_global) * 4 + color];
        }
        else
        {
            shared_block[(y * shared_block_width + x) * BLOCK_Z + color] = 0;
        }

        sub_pixel_idx += BLOCK_X * BLOCK_Y;
    }

    __syncthreads();

    //set alpha to max value
    if (color == 3)
    {
        out[idx] = 255;
    }
    else
    {
        float sum = 0.0;

        for (int i = 0; i < CONV_SIDE_LENGTH; i++)
        {
            for (int j = 0; j < CONV_SIDE_LENGTH; j++)
            {
                int idx = ((threadIdx.y + i) * shared_block_width + threadIdx.x + j) * BLOCK_Z + color;
                sum += constantConv[i * CONV_SIDE_LENGTH + j] * shared_block[idx];
            }
        }
        
        out[idx] = clamp(sum);
    }


}


__global__ void ConvolveSharedMemory1D(unsigned char* out, unsigned char* pixels, int  width, int height)
{

    //int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    //if (global_threadIdx >= width * height * BLOCK_Z) return;

    //int convOffset = CONV_SIDE_LENGTH / 2;

    //const int shared_block_width =  BLOCK_X + CONV_SIDE_LENGTH - 1;
    //const int shared_block_height = BLOCK_Y + CONV_SIDE_LENGTH - 1;
    //const int shared_block_size = shared_block_width * shared_block_height * BLOCK_Z;

    //int color = global_threadIdx % 4;
    //__shared__ unsigned char shared_block[shared_block_size];

    ////set shared memory
    //int sub_idx = threadIdx.x;

    //while (sub_idx < shared_block_size)
    //{
    //    int x = (sub_idx / 4) % shared_block_width;
    //    int y = (sub_idx / 4) / shared_block_width;

    //    int x_global = (x - convOffset) + blockIdx.x * BLOCK_X;
    //    int y_global = (y - convOffset) + blockIdx.y * BLOCK_Y;

    //    if (x_global >= 0 && y_global >= 0 && x_global < width && y_global < height)
    //    {
    //        shared_block[(x * shared_block_width + y) + color] = pixels[];
    //    }
    //    else
    //    {
    //        shared_block[sub_idx + color] = 0;
    //    }

    //    sub_idx += shared_block_size;
    //}

    //__syncthreads();

    ////set alpha to full value
    //if (color == 3)
    //{
    //    out[global_threadIdx] = 255;
    //}
    //else
    //{
    //    float sum = 0.0;

    //    int block_row = threadIdx.x / shared_block_width;
    //    int block_col = threadIdx.x / 
    //    int location_idx = threadIdx.x / 4;
    //    for (int i = 0; i < CONV_SIDE_LENGTH; i++)
    //    {
    //        for (int j = 0; j < CONV_SIDE_LENGTH; j++)
    //        {
    //            sum += constantConv[i * CONV_SIDE_LENGTH + j] * shared_block[];
    //        }
    //    }

    //    out[global_threadIdx] = clamp(sum);
    //}

}

vector<unsigned char> ConvolveImage(vector<unsigned char>& pixels, vector<float>& convolution, int width, int height, int convHeight, int convWidth)
{
    unsigned char* input;
    unsigned char* out;
    //float* cudaConv;


    int pixelCount = pixels.size();
    int pixelsMemory = sizeof(unsigned char) * pixelCount;

    check(cudaMalloc((void**)&input, pixelsMemory));
    check(cudaMalloc((void**)&out, pixelsMemory));
    //check(cudaMalloc((void**)&cudaConv, convHeight * convWidth * sizeof(float)));

    //check(cudaMemcpy(cudaConv, convolution.data(), convHeight * convWidth * sizeof(float), cudaMemcpyHostToDevice));

    check(cudaMemcpy(input, pixels.data(), pixelsMemory, cudaMemcpyHostToDevice));

    dim3 pixelGrid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
    dim3 subGrid(BLOCK_X, BLOCK_Y, BLOCK_Z);

    const size_t num_iter = 10;
    float totalTime = 0.0f;
    for (size_t i = 0; i < num_iter; i++)
    {
        CudaTiming kernelTiming;
        kernelTiming.Start();
        ConvolveSharedMemory << < pixelGrid, subGrid >> > (out, input, width, height);
        //Convolve << < pixelGrid, subGrid >> > (out, input, width, height);
        cudaProfilerStop();
        kernelTiming.Stop();
        totalTime += kernelTiming.GetTime();
    }

    std::cout << "Avg time: " << totalTime / (float)num_iter << " ms\n";


    unsigned char* outputPointer = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));
    check(cudaMemcpy(outputPointer, out, pixelCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    check(cudaFree(input));
    //check(cudaFree(cudaConv));

    vector<unsigned char> outputPixels;
    outputPixels.reserve(pixelCount);

    for (int i = 0; i < pixelCount; i++)
    {
        outputPixels.push_back(outputPointer[i]);
    }

    free(outputPointer);
    check(cudaFree(out));

    return outputPixels;
}
int main(void) {
    vector<unsigned char> pixels;
    unsigned int width = 1024;
    unsigned int height = 1024;
    lodepng::decode(pixels, width, height, "cat_image.png");


    vector<float> convolution(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH, 0);

    for (int i = 0; i < convolution.size(); i++)
    {
        convolution[i] = (2.0f / (float)(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH)) * (float)rand() / (float)RAND_MAX;
    }
    //identity
    //convolution[(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH) / 2 ] = 1.1;

    //convolution[25] = 2.3;
    //vector<float> convolution = { -1,-1,-1,-1,8,-1,-1,-1,-1 };

   
    check(cudaMemcpyToSymbol(constantConv, convolution.data(), convolution.size() * sizeof(float)));

    //CudaTiming ct;
   // ct.Start();
    vector<unsigned char> newImage = ConvolveImage(pixels, convolution, width, height, CONV_SIDE_LENGTH, CONV_SIDE_LENGTH);
    //ct.Stop();
    //ct.PrintTime("Total function time");

    lodepng::encode("conv_image.png", newImage, width, height);

}