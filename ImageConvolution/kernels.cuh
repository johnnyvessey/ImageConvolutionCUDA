#include "CudaTiming.cuh"
#include <vector>

using std::vector;

#pragma once

#define CONV_SIDE_LENGTH 15

#define BLOCK_X 16
#define BLOCK_Y 8
#define BLOCK_Z 4


__device__ unsigned char clamp(float sum);

__global__ void NaiveConvolve(unsigned char* out, unsigned char* pixels, float* cudaConv, int width, int height);

__global__ void ConvolveSharedMemory(unsigned char* out, const unsigned char* pixels, const int width, const int height);

__global__ void ConvolveSharedMemoryNoZ(unsigned char* out, const unsigned char* pixels, const int width, const int height);

vector<unsigned char> ConvolveImage(vector<unsigned char>& pixels, vector<float>& convolution, int width, int height, int convHeight, int convWidth, bool naive);


//__global__ void ConvolveSharedMemoryAlign(unsigned char* out, const unsigned char* pixels, const int width, const int height)
//{
//
//    int row = blockIdx.z * BLOCK_Z + threadIdx.z;
//
//    int col = blockIdx.y * BLOCK_Y + threadIdx.y;
//    int color = threadIdx.x;
//
//    int idx = (row * width + col) * 4 + color;
//    if (idx >= width * height * 4) return;
//
//    int convOffset = CONV_SIDE_LENGTH / 2;
//
//    const int shared_block_width = BLOCK_Y + CONV_SIDE_LENGTH - 1;
//    const int shared_block_height = BLOCK_Z + CONV_SIDE_LENGTH - 1;
//    const int shared_block_size = shared_block_width * shared_block_height;
//
//    __shared__ unsigned char shared_block[shared_block_height * shared_block_width * 4];
//
//    //set shared memory
//    int sub_pixel_idx = threadIdx.z * blockDim.y + threadIdx.y;
//
//    while (sub_pixel_idx < shared_block_size)
//    {
//        int c = sub_pixel_idx % shared_block_width;
//        int r = sub_pixel_idx / shared_block_width;
//
//        int col_global = (c - convOffset) + blockIdx.y * BLOCK_Y;
//        int row_global = (r - convOffset) + blockIdx.z * BLOCK_Z;
//
//        if (col_global >= 0 && row_global >= 0 && col_global < width && row_global < height)
//        {
//            shared_block[(r * shared_block_width + c) * 4 + color] = pixels[(row_global * width + col_global) * 4 + color];
//        }
//        else
//        {
//            shared_block[(r * shared_block_width + c) * 4 + color] = 0;
//        }
//
//        sub_pixel_idx += BLOCK_Z * BLOCK_Y;
//    }
//
//    __syncthreads();
//
//    float sum = 0.0;
//
//    for (int i = 0; i < CONV_SIDE_LENGTH; i++)
//    {
//        for (int j = 0; j < CONV_SIDE_LENGTH; j++)
//        {
//            int idx = ((threadIdx.z + i) * shared_block_width + threadIdx.y + j) * 4 + color;
//            sum += constantConv[i * CONV_SIDE_LENGTH + j] * shared_block[idx];
//        }
//    }
//
//    out[idx] = clamp(sum);
//
//}
//
//__global__ void ConvolveSharedMemory1D(unsigned char* out, unsigned char* pixels, int  width, int height)
//{
//
//    //int global_threadIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 4 + threadIdx.y;
//    //int blockThreadOffset = blockIdx.x * blockDim.x;
//    //int color = threadIdx.y;
//    //int pixelsSize = width * height * 4;
//    //if (global_threadIdx >= width * height * BLOCK_Z) return;
//
//    //int convOffset = CONV_SIDE_LENGTH / 2;
//
//    //const int shared_block_width = 64; // BLOCK_X + CONV_SIDE_LENGTH - 1;
//    //const int shared_block_height = 64; // BLOCK_Y + CONV_SIDE_LENGTH - 1;
//    //const int shared_block_size = shared_block_width * shared_block_height * 4;
//
//    //__shared__ unsigned char shared_block[shared_block_size][4];
//
//    ////set shared memory
//    //int sub_idx = threadIdx.x;
//
//    //while (sub_idx < shared_block_size)
//    //{
//    //    int global_idx = 4 * (sub_idx + blockThreadOffset) + color;
//    //    if (global_idx >= 0 && global_idx < pixelsSize)
//    //    {
//    //        shared_block[sub_idx][color] = pixels[global_idx];
//    //    }
//    //    else
//    //    {
//    //        shared_block[sub_idx][color] = 0;
//    //    }
//
//    //    sub_idx += blockDim.x;
//    //}
//
//    //__syncthreads();
//
//    ////set alpha to full value
//    //if (color == 3)
//    //{
//    //    out[global_threadIdx] = 255;
//    //}
//    //else
//    //{
//    //    float sum = 0.0;
//    //    
//    //    int sub_idx = threadIdx.x;
//    //    while (sub_idx < shared_block_size)
//    //    {
//    //        for (int i = 0; i < CONV_SIDE_LENGTH; i++)
//    //        {
//    //            for (int j = 0; j < CONV_SIDE_LENGTH; j++)
//    //            {
//    //                sum += constantConv[i * CONV_SIDE_LENGTH + j] * shared_block[sub_idx + i * 64 + j][color];
//    //            }
//    //        }
//
//    //        out[global_threadIdx] = clamp(sum);      
//    //        sub_idx += blockDim.x;
//    //    }
//
//    //    
//
//    //}
//
//}
