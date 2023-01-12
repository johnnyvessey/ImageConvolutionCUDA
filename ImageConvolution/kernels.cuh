#include "CudaTiming.cuh"
#include <vector>
#include <chrono>
#include <cuda_fp16.h>
using namespace std::chrono;
using std::vector;

#pragma once

#define CONV_SIDE_LENGTH 41

#define BLOCK_X 16
#define BLOCK_Y 8

#define BLOCK_Z 4

#define BLOCK_X_UNSEPARABLE 16
#define BLOCK_Y_UNSEPARABLE 8

#define EPSILON .00001f

#define H_BLOCK_X 1024

#define V_BLOCK_X 16
#define V_BLOCK_Y 32

#define H_BLOCK_X 128

#define IMAGE_WIDTH 2048
#define IMAGE_HEIGHT 2048

#define TRANSPOSE_BLOCK_SIDE 32


#define BLOCK_X_4_PIXEL 16
#define BLOCK_Y_4_PIXEL 8

struct pixel
{
    unsigned char R;
    unsigned char G;
    unsigned char B;
    unsigned char A;
};

struct pixel4
{
    pixel pixels[4];
};




class ImageConvolution
{
public:

	static unsigned char* ConvolveImage(vector<unsigned char>& pixels, vector<float>& convolution, int width, int height, int convWidth, int convHeight, bool naive, bool useConstantMemory);

	static unsigned char* ConvolveOptimized(vector<unsigned char>& pixels, vector<float>& convolution, int width, int height, int convWidth, int convHeight, bool useChar = false);

};
