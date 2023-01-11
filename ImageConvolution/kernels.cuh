#include "CudaTiming.cuh"
#include <vector>
#include <chrono>

using namespace std::chrono;
using std::vector;

#pragma once

#define CONV_SIDE_LENGTH 99

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

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024

#define TRANSPOSE_BLOCK_SIDE 32


class ImageConvolution
{
public:

	static unsigned char* ConvolveImage(vector<unsigned char>& pixels, vector<float>& convolution, int width, int height, int convWidth, int convHeight, bool naive);

	static unsigned char* ConvolveOptimized(vector<unsigned char>& pixels, vector<float>& convolution, int width, int height, int convWidth, int convHeight);

};
