#include "kernels.cuh"
#include "lodepng.h"

__constant__ float constantConv[CONV_SIDE_LENGTH * CONV_SIDE_LENGTH];
__constant__ float constantConv1d[CONV_SIDE_LENGTH];


__device__ unsigned char clamp(float sum)
{
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

namespace Utils
{
    __global__ void Transpose(unsigned char* outT, unsigned char* out)
    {
        __shared__ unsigned char tile[TRANSPOSE_BLOCK_SIDE][TRANSPOSE_BLOCK_SIDE][4];

        int tx = threadIdx.x, ty = threadIdx.y;
        int col = blockIdx.x * blockDim.x + tx;
        int row = blockIdx.y * blockDim.y + ty;

        tile[ty][tx][0] = out[4 * (row * IMAGE_WIDTH + col)];
        tile[ty][tx][1] = out[4 * (row * IMAGE_WIDTH + col) + 1];
        tile[ty][tx][2] = out[4 * (row * IMAGE_WIDTH + col) + 2];
        tile[ty][tx][3] = out[4 * (row * IMAGE_WIDTH + col) + 3];

        __syncthreads();

        outT[4 * ((col * IMAGE_WIDTH) + row)] = tile[ty][tx][0];
        outT[4 * ((col * IMAGE_WIDTH) + row) + 1] = tile[ty][tx][1];
        outT[4 * ((col * IMAGE_WIDTH) + row) + 2] = tile[ty][tx][2];
        outT[4 * ((col * IMAGE_WIDTH) + row) + 3] = tile[ty][tx][3];
    }

    void SaveToImage(unsigned char* out, const char* filename)
    {
        int size = IMAGE_WIDTH * IMAGE_HEIGHT * 4;
        unsigned char* imagePixels = (unsigned char*)malloc(size * sizeof(unsigned char));
        cudaMemcpy(imagePixels, out, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        vector<unsigned char> pixels;
        pixels.reserve(size);
        for (size_t i = 0; i < size; i += 4)
        {
            pixels.push_back(imagePixels[i]);
            pixels.push_back(imagePixels[i + 1]);
            pixels.push_back(imagePixels[i + 2]);
            pixels.push_back(255);

        }

        free(imagePixels);
        lodepng::encode(filename, pixels, IMAGE_WIDTH, IMAGE_HEIGHT);
    }

};

namespace ConvolutionSeparation
{
    bool isConvSeparable(vector<float>& conv, int convWidth, int convHeight, vector<float>& h_conv, vector<float>& v_conv)
    {
        h_conv = vector<float>(conv.begin(), conv.begin() + convWidth);
        v_conv.resize(convHeight);

        float firstElem = conv[0];
        v_conv[0] = 1.0f;

        for (size_t row = 1; row < convHeight; row++)
        {
            float rowRatio = conv[row * convWidth] != 0 ? conv[row * convWidth] / firstElem : 0;

            for (size_t col = 1; col < convWidth; col++)
            {
                if (abs(conv[row * convWidth + col] - rowRatio * h_conv[col]) > EPSILON)
                {
                    return false;
                }
            }
            v_conv[row] = rowRatio;

        }
        return true;

    }

    __global__ void isConvSeparableGPU(bool* isSeparable, float* convolution, float* v_conv)
    {
        __shared__ float rowRatio;

        if (threadIdx.x == 0)
        {
            float convFirstElem = convolution[0];
            rowRatio = (convFirstElem != 0) ? convolution[blockIdx.x * CONV_SIDE_LENGTH] / convFirstElem : 0;
        }
        __syncthreads();

        float value = convolution[threadIdx.x];
        int global_idx = blockIdx.x * CONV_SIDE_LENGTH + threadIdx.x;
        bool matches = abs(convolution[global_idx] - rowRatio * convolution[threadIdx.x]) < EPSILON;

        if (!matches)
        {
            *isSeparable = false;
        }

        if (threadIdx.x == 0)
        {
            v_conv[blockIdx.x] = rowRatio;
        }

    }

};

namespace Naive
{
    __global__ void NaiveConvolveConstantMemory(unsigned char* out, unsigned char* pixels, float* cudaConv, int width, int height)
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

    __global__ void NaiveConvolveNoConstantMemory(unsigned char* out, unsigned char* pixels, float* cudaConv, int width, int height)
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
                sum = sum + (cudaConv[convIdx] * pixelVal);


            }
        }
        int idx = (row * width + col) * 4 + color;


        out[idx] = clamp(sum);


    }

};

namespace BasicSharedMemory
{
    __global__ void ConvolveSharedMemory(unsigned char* out, const unsigned char* pixels, const int width, const int height)
    {

        int row = blockIdx.y * BLOCK_Y + threadIdx.y;
        int col = blockIdx.x * BLOCK_X + threadIdx.x;
        int color = threadIdx.z;

        int global_idx = (row * width + col) * 4 + color;
        if (global_idx >= width * height * BLOCK_Z) return;

        int convOffset = CONV_SIDE_LENGTH / 2;

        const int shared_block_width = BLOCK_X + CONV_SIDE_LENGTH - 1;
        const int shared_block_height = BLOCK_Y + CONV_SIDE_LENGTH - 1;
        const int shared_block_size = shared_block_width * shared_block_height;

        __shared__ unsigned char shared_block[shared_block_height * shared_block_width * BLOCK_Z];

        //set shared memory

        for (int sub_pixel_idx = threadIdx.y * blockDim.x + threadIdx.x; sub_pixel_idx < shared_block_size; sub_pixel_idx += (BLOCK_X * BLOCK_Y))
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

        }

        __syncthreads();

        float sum = 0.0;
        for (int i = 0; i < CONV_SIDE_LENGTH; i++)
        {
            for (int j = 0; j < CONV_SIDE_LENGTH; j++)
            {
                int shared_block_idx = ((threadIdx.y + i) * shared_block_width + threadIdx.x + j) * BLOCK_Z + color;
                sum += constantConv[i * CONV_SIDE_LENGTH + j] * shared_block[shared_block_idx];
            }
        }

        out[global_idx] = clamp(sum);

    }

    __global__ void ConvolveSharedMemoryNoZ(unsigned char* out, const unsigned char* pixels, const int width, const int height)
    {

        int row = blockIdx.y * BLOCK_Y + threadIdx.y;
        int col = blockIdx.x * BLOCK_X + threadIdx.x;

        int idx = (row * width + col);
        if (idx >= width * height) return;

        int convOffset = CONV_SIDE_LENGTH / 2;

        const int shared_block_width = BLOCK_X + CONV_SIDE_LENGTH - 1;
        const int shared_block_height = BLOCK_Y + CONV_SIDE_LENGTH - 1;
        const int shared_block_size = shared_block_width * shared_block_height;

        __shared__ unsigned char shared_block[shared_block_height * shared_block_width * 4];

        //set shared memory

        for (int sub_pixel_idx = threadIdx.y * blockDim.x + threadIdx.x; sub_pixel_idx < shared_block_size; sub_pixel_idx += (BLOCK_X * BLOCK_Y))
        {
            int x = sub_pixel_idx % shared_block_width;
            int y = sub_pixel_idx / shared_block_width;

            int x_global = (x - convOffset) + blockIdx.x * BLOCK_X;
            int y_global = (y - convOffset) + blockIdx.y * BLOCK_Y;

            if (x_global >= 0 && y_global >= 0 && x_global < width && y_global < height)
            {

                for (int color = 0; color < 4; color++)
                {
                    shared_block[(y * shared_block_width + x) * 4 + color] = pixels[(y_global * width + x_global) * 4 + color];
                }
            }
            else
            {
                for (int color = 0; color < 4; color++)
                {
                    shared_block[(y * shared_block_width + x) * 4 + color] = 0;
                }
            }

        }

        __syncthreads();

        for (int color = 0; color < 4; color++)
        {
            float sum = 0.0;
            for (int i = 0; i < CONV_SIDE_LENGTH; i++)
            {
                for (int j = 0; j < CONV_SIDE_LENGTH; j++)
                {
                    int shared_idx = ((threadIdx.y + i) * shared_block_width + threadIdx.x + j) * 4 + color;
                    sum += constantConv[i * CONV_SIDE_LENGTH + j] * shared_block[shared_idx];

                }
            }
            out[4 * idx + color] = clamp(sum);

        }



    }

    __global__ void ConvolveHorizontalRowPerBlock(float* out, const unsigned char* pixels)
    {
        __shared__ unsigned char row[IMAGE_WIDTH * 4];

        //optimized for up to 1024 width images; use loop for wider images
        int tx = threadIdx.x;
        int global_idx = 4 * (blockIdx.x * IMAGE_WIDTH + tx);

        //loop unrolling optimization
        row[4 * tx] = pixels[global_idx];
        row[4 * tx + 1] = pixels[global_idx + 1];
        row[4 * tx + 2] = pixels[global_idx + 2];
        row[4 * tx + 3] = pixels[global_idx + 3];
        __syncthreads();

        int halfConvolve = CONV_SIDE_LENGTH / 2;

        float sumR = 0, sumG = 0, sumB = 0, sumA = 0;
        for (int i = -halfConvolve; i <= halfConvolve; i++)
        {
            int idx = 4 * (tx + i);
            if (idx >= 0 && idx < IMAGE_WIDTH * 4)
            {
                float conv = constantConv1d[i + halfConvolve];

                sumR += (conv * row[idx]);
                sumG += (conv * row[idx + 1]);
                sumB += (conv * row[idx + 2]);
                sumA += (conv * row[idx + 3]);
            }
        }


        out[global_idx] = sumR;
        out[global_idx + 1] = sumG;
        out[global_idx + 2] = sumB;
        out[global_idx + 3] = sumA;


    }
};

namespace Optimized
{
    __global__ void ConvolveHorizontalSplitUp(half* out, const unsigned char* __restrict__ pixels)
    {
        int global_row = blockIdx.y;
        int global_col = blockIdx.x * H_BLOCK_X + threadIdx.x;

        const int conv_offset = CONV_SIDE_LENGTH / 2;
        const int arrSize = 4 * IMAGE_WIDTH * IMAGE_HEIGHT;

        int global_idx = 4 * (global_row * IMAGE_WIDTH + global_col);
        if (global_idx >= arrSize) return;

        //faster if using float for shared memory (32 bit data type)
        __shared__ float tile[4 * (H_BLOCK_X + CONV_SIDE_LENGTH - 1)];

        int col = threadIdx.x;
        int globalOffset = -4 * conv_offset;

        while (col < H_BLOCK_X + CONV_SIDE_LENGTH - 1)
        {
            int cur_global_idx = global_idx + globalOffset;
            bool in_bounds = (global_col - conv_offset) >= 0 && (global_col - conv_offset) < IMAGE_WIDTH;
            if (in_bounds)
            {
                tile[4 * col] = pixels[cur_global_idx];
                tile[4 * col + 1] = pixels[cur_global_idx + 1];
                tile[4 * col + 2] = pixels[cur_global_idx + 2];
                tile[4 * col + 3] = pixels[cur_global_idx + 3];
            }
            else
            {
                tile[4 * col] = 0;
                tile[4 * col + 1] = 0;
                tile[4 * col + 2] = 0;
                tile[4 * col + 3] = 0;
            }


            col += H_BLOCK_X;
            global_col += H_BLOCK_X;
            globalOffset += (4 * H_BLOCK_X);
        }

        __syncthreads();

        float sumR = 0, sumG = 0, sumB = 0, sumA = 0;

        for (int i = 0; i < CONV_SIDE_LENGTH; i++)
        {
            float conv = constantConv1d[i];

            int tile_col = 4 * (threadIdx.x + i);

            sumR += (conv * tile[tile_col]);
            sumG += (conv * tile[tile_col + 1]);
            sumB += (conv * tile[tile_col + 2]);
            sumA += (conv * tile[tile_col + 3]);

        }

        out[global_idx] = __float2half(sumR);
        out[global_idx + 1] = __float2half(sumG);
        out[global_idx + 2] = __float2half(sumB);
        out[global_idx + 3] = __float2half(sumA);


    }

    __global__ void ConvolveVertical(unsigned char* out, const half* __restrict__ pixels)
    {
        int global_row = blockIdx.y * V_BLOCK_Y + threadIdx.y;
        int global_col = blockIdx.x * V_BLOCK_X + threadIdx.x;

        const int conv_offset = CONV_SIDE_LENGTH / 2;
        const int arrSize = 4 * IMAGE_WIDTH * IMAGE_HEIGHT;

        int global_idx = 4 * (global_row * IMAGE_WIDTH + global_col);
        if (global_idx >= arrSize) return;

        __shared__ half tile[V_BLOCK_Y + CONV_SIDE_LENGTH - 1][4 * V_BLOCK_X];

        int row = threadIdx.y;
        int globalOffset = -4 * IMAGE_WIDTH * conv_offset;

        int tx4 = threadIdx.x * 4;
        while (row < V_BLOCK_Y + CONV_SIDE_LENGTH - 1)
        {
            int cur_global_idx = global_idx + globalOffset;
            bool in_bounds = cur_global_idx >= 0 && cur_global_idx < arrSize;
            if (in_bounds)
            {
                tile[row][tx4] = pixels[cur_global_idx];
                tile[row][tx4 + 1] = pixels[cur_global_idx + 1];
                tile[row][tx4 + 2] = pixels[cur_global_idx + 2];
                tile[row][tx4 + 3] = pixels[cur_global_idx + 3];
            }
            else
            {
                tile[row][tx4] =  0.0f;
                tile[row][tx4 + 1] = 0.0f;
                tile[row][tx4 + 2] = 0.0f;
                tile[row][tx4 + 3] = 0.0f;
            }


            row += V_BLOCK_Y;
            globalOffset += (V_BLOCK_Y * 4 * IMAGE_WIDTH);
        }

        __syncthreads();

        half sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumA = 0.0f;

        for (int i = 0; i < CONV_SIDE_LENGTH; i++)
        {
            half conv = __float2half(constantConv1d[i]);

            int tile_row = threadIdx.y;
            int tile_col = threadIdx.x * 4;

            sumR = sumR + (conv * tile[tile_row + i][tile_col]);
            sumG = sumG +(conv * tile[tile_row + i][tile_col + 1]);
            sumB = sumB + (conv * tile[tile_row + i][tile_col + 2]);
            sumA = sumA + (conv * tile[tile_row + i][tile_col + 3]);

        }

        out[global_idx] = clamp(sumR);
        out[global_idx + 1] = clamp(sumG);
        out[global_idx + 2] = clamp(sumB);
        out[global_idx + 3] = clamp(sumA);
    }

    __global__ void ConvolveSharedMemoryUnseparableOptimizedChar(unsigned char* out, const unsigned char* __restrict__ pixels)
    {

        int row = blockIdx.y * BLOCK_Y_UNSEPARABLE + threadIdx.y;
        int col = blockIdx.x * BLOCK_X_UNSEPARABLE + threadIdx.x;

        int global_idx = (row * IMAGE_WIDTH + col) * 4;
        if (global_idx >= IMAGE_WIDTH * IMAGE_HEIGHT * 4) return;

        int convOffset = CONV_SIDE_LENGTH / 2;

        const int shared_block_width = BLOCK_X_UNSEPARABLE + CONV_SIDE_LENGTH - 1;
        const int shared_block_height = BLOCK_Y_UNSEPARABLE + CONV_SIDE_LENGTH - 1;
        const int shared_block_size = shared_block_width * shared_block_height;

        __shared__ unsigned char shared_block[shared_block_size * 4];

        //set shared memory

        for (int sub_pixel_idx = threadIdx.y * BLOCK_X_UNSEPARABLE + threadIdx.x; sub_pixel_idx < shared_block_size; sub_pixel_idx += (BLOCK_X_UNSEPARABLE * BLOCK_Y_UNSEPARABLE))
        {
            int x = sub_pixel_idx % shared_block_width;
            int y = sub_pixel_idx / shared_block_width;

            int x_global = (x - convOffset) + blockIdx.x * BLOCK_X_UNSEPARABLE;
            int y_global = (y - convOffset) + blockIdx.y * BLOCK_Y_UNSEPARABLE;

            if (x_global >= 0 && y_global >= 0 && x_global < IMAGE_WIDTH && y_global < IMAGE_HEIGHT)
            {
                shared_block[(y * shared_block_width + x) * 4] = pixels[(y_global * IMAGE_WIDTH + x_global) * 4];
                shared_block[(y * shared_block_width + x) * 4 + 1] = pixels[(y_global * IMAGE_WIDTH + x_global) * 4 + 1];
                shared_block[(y * shared_block_width + x) * 4 + 2] = pixels[(y_global * IMAGE_WIDTH + x_global) * 4 + 2];
                shared_block[(y * shared_block_width + x) * 4 + 3] = pixels[(y_global * IMAGE_WIDTH + x_global) * 4 + 3];
            }
            else
            {
                shared_block[(y * shared_block_width + x) * 4] = 0;
                shared_block[(y * shared_block_width + x) * 4 + 1] = 0;
                shared_block[(y * shared_block_width + x) * 4 + 2] = 0;
                shared_block[(y * shared_block_width + x) * 4 + 3] = 0;
            }

        }

        __syncthreads();

        float sumR = 0.0, sumG = 0.0, sumB = 0.0, sumA = 0.0;
        for (int i = 0; i < CONV_SIDE_LENGTH; i++)
        {
            for (int j = 0; j < CONV_SIDE_LENGTH; j++)
            {
                float conv = constantConv[i * CONV_SIDE_LENGTH + j];
                int shared_block_idx = ((threadIdx.y + i) * shared_block_width + threadIdx.x + j) * 4;

                sumR += conv * shared_block[shared_block_idx];
                sumG += conv * shared_block[shared_block_idx + 1];
                sumB += conv * shared_block[shared_block_idx + 2];
                sumA += conv * shared_block[shared_block_idx + 3];

            }
        }

        out[global_idx] = clamp(sumR);
        out[global_idx + 1] = clamp(sumG);
        out[global_idx + 2] = clamp(sumB);
        out[global_idx + 3] = clamp(sumA);
    }

    __global__ void ConvolveSharedMemoryUnseparableOptimizedInt(unsigned char* out, const unsigned char* __restrict__ pixels)
    {

        int row = blockIdx.y * BLOCK_Y_UNSEPARABLE + threadIdx.y;
        int col = blockIdx.x * BLOCK_X_UNSEPARABLE + threadIdx.x;

        int global_idx = (row * IMAGE_WIDTH + col) * 4;
        if (global_idx >= IMAGE_WIDTH * IMAGE_HEIGHT * 4) return;

        int convOffset = CONV_SIDE_LENGTH / 2;

        const int shared_block_width = BLOCK_X_UNSEPARABLE + CONV_SIDE_LENGTH - 1;
        const int shared_block_height = BLOCK_Y_UNSEPARABLE + CONV_SIDE_LENGTH - 1;
        const int shared_block_size = shared_block_width * shared_block_height;

        __shared__ int shared_block[shared_block_size * 4];

        //set shared memory

        for (int sub_pixel_idx = threadIdx.y * BLOCK_X_UNSEPARABLE + threadIdx.x; sub_pixel_idx < shared_block_size; sub_pixel_idx += (BLOCK_X_UNSEPARABLE * BLOCK_Y_UNSEPARABLE))
        {
            int x = sub_pixel_idx % shared_block_width;
            int y = sub_pixel_idx / shared_block_width;

            int x_global = (x - convOffset) + blockIdx.x * BLOCK_X_UNSEPARABLE;
            int y_global = (y - convOffset) + blockIdx.y * BLOCK_Y_UNSEPARABLE;

            if (x_global >= 0 && y_global >= 0 && x_global < IMAGE_WIDTH && y_global < IMAGE_HEIGHT)
            {
                shared_block[(y * shared_block_width + x) * 4] = pixels[(y_global * IMAGE_WIDTH + x_global) * 4];
                shared_block[(y * shared_block_width + x) * 4 + 1] = pixels[(y_global * IMAGE_WIDTH + x_global) * 4 + 1];
                shared_block[(y * shared_block_width + x) * 4 + 2] = pixels[(y_global * IMAGE_WIDTH + x_global) * 4 + 2];
                shared_block[(y * shared_block_width + x) * 4 + 3] = pixels[(y_global * IMAGE_WIDTH + x_global) * 4 + 3];
            }
            else
            {
                shared_block[(y * shared_block_width + x) * 4] = 0;
                shared_block[(y * shared_block_width + x) * 4 + 1] = 0;
                shared_block[(y * shared_block_width + x) * 4 + 2] = 0;
                shared_block[(y * shared_block_width + x) * 4 + 3] = 0;
            }

        }

        __syncthreads();

        float sumR = 0.0, sumG = 0.0, sumB = 0.0, sumA = 0.0;
        for (int i = 0; i < CONV_SIDE_LENGTH; i++)
        {
            for (int j = 0; j < CONV_SIDE_LENGTH; j++)
            {
                float conv = constantConv[i * CONV_SIDE_LENGTH + j];
                int shared_block_idx = ((threadIdx.y + i) * shared_block_width + threadIdx.x + j) * 4;

                sumR += conv * shared_block[shared_block_idx];
                sumG += conv * shared_block[shared_block_idx + 1];
                sumB += conv * shared_block[shared_block_idx + 2];
                sumA += conv * shared_block[shared_block_idx + 3];

            }
        }

        out[global_idx] = clamp(sumR);
        out[global_idx + 1] = clamp(sumG);
        out[global_idx + 2] = clamp(sumB);
        out[global_idx + 3] = clamp(sumA);
    }


};

namespace Experimental
{
    __device__ pixel4 zeroPixel4()
    {
        pixel4 p4;
#pragma unroll 4
        for (int i = 0; i < 4; i++)
        {
            pixel p; p.R = 0; p.G = 0; p.B = 0; p.A = 0;
            p4.pixels[i] = p;
        }
        return p4;
    }

    __global__ void ConvolveSharedMemoryUnseparableOptimizedPixel(pixel* out, const pixel4* __restrict__ pixels)
    {

        int row = blockIdx.y * BLOCK_Y_4_PIXEL + threadIdx.y;
        int col = (blockIdx.x * BLOCK_X_4_PIXEL + threadIdx.x);
        const int width = BLOCK_X_4_PIXEL + (CONV_SIDE_LENGTH - 1) / 4;

        int global_idx = (row * IMAGE_WIDTH + 4 * col);
        if (global_idx >= IMAGE_WIDTH * IMAGE_HEIGHT) return;

        int convOffset = CONV_SIDE_LENGTH / 2;

        const int shared_block_width = (4 * BLOCK_X_4_PIXEL + CONV_SIDE_LENGTH - 1);
        const int shared_block_height = BLOCK_Y_4_PIXEL + CONV_SIDE_LENGTH - 1;
        const int shared_block_size = shared_block_width * shared_block_height;
        const int image_width = IMAGE_WIDTH / 4;


        __shared__ pixel4 shared_block[shared_block_height][width];

        //set shared memory

        int sub_pixel_idx = BLOCK_X_4_PIXEL * threadIdx.y + threadIdx.x;

        while (sub_pixel_idx < width * shared_block_height)
        {
            int x_idx = sub_pixel_idx % width;
            int y_idx = sub_pixel_idx / width;
            int y_global = (y_idx - convOffset) + blockIdx.y * BLOCK_Y_4_PIXEL;
            int x_global = (x_idx - convOffset) + blockIdx.x * BLOCK_X_4_PIXEL;



            if (x_global >= 0 && y_global >= 0 && x_global < image_width && y_global < IMAGE_HEIGHT)
            {
                shared_block[y_idx][x_idx] = pixels[y_global * image_width + x_global];
            }
            else {
                shared_block[y_idx][x_idx] = zeroPixel4();
            }

            sub_pixel_idx += (BLOCK_Y_4_PIXEL * BLOCK_X_4_PIXEL);
        }

        __syncthreads();

        float sumR[4] = { 0,0,0,0 };
        float sumG[4] = { 0,0,0,0 };
        float sumB[4] = { 0,0,0,0 };
        float sumA[4] = { 0,0,0,0 };


        for (int i = 0; i < CONV_SIDE_LENGTH; i++)
        {
            for (int j = 0; j < CONV_SIDE_LENGTH; j++)
            {
                float conv = constantConv[i * CONV_SIDE_LENGTH + j];

                pixel pixel_buffer[8];

                pixel4 left = shared_block[threadIdx.y + i][threadIdx.x + j / 4];
#pragma unroll 4
                for (int b_idx = 0; b_idx < 4; b_idx++)
                {
                    pixel_buffer[b_idx] = left.pixels[b_idx];
                }
                pixel4 right;
                if (j % 4 != 0)
                {
                    right = shared_block[threadIdx.y + i][threadIdx.x + j / 4 + 1];
#pragma unroll 4
                    for (int b_idx = 4; b_idx < 8; b_idx++)
                    {
                        pixel_buffer[b_idx] = right.pixels[b_idx - 4];
                    }
                }

#pragma unroll 4
                for (int sub_thread = 0; sub_thread < 4; sub_thread++)
                {
                    sumR[sub_thread] += conv * pixel_buffer[sub_thread + (j % 4)].R;
                    sumG[sub_thread] += conv * pixel_buffer[sub_thread + (j % 4)].G;
                    sumB[sub_thread] += conv * pixel_buffer[sub_thread + (j % 4)].B;
                    sumA[sub_thread] += conv * pixel_buffer[sub_thread + (j % 4)].A;

                }

            }
        }



#pragma unroll 4
        for (int i = 0; i < 4; i++)
        {
            pixel p;
            p.R = clamp(sumR[i]);
            p.G = clamp(sumG[i]);
            p.B = clamp(sumB[i]);
            p.A = clamp(sumA[i]);
            out[global_idx + i] = p;
        }

    }

};

unsigned char* ImageConvolution::ConvolveImage(vector<unsigned char>& pixels, vector<float>& convolution, int width, int height, int convWidth, int convHeight, bool naive, bool useConstantMemory)
{
    check(cudaMemcpyToSymbol(constantConv, convolution.data(), convolution.size() * sizeof(float)));

    unsigned char* input;
    unsigned char* out;


    int pixelCount = pixels.size();
    int pixelsMemory = sizeof(unsigned char) * pixelCount;

    check(cudaMalloc((void**)&input, pixelsMemory));
    check(cudaMalloc((void**)&out, pixelsMemory));
    check(cudaMemcpy(input, pixels.data(), pixelsMemory, cudaMemcpyHostToDevice));

    dim3 pixelGrid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
    dim3 subGrid(BLOCK_X, BLOCK_Y, BLOCK_Z);


    if (naive)
    {
        float* cudaConv;
        check(cudaMalloc((void**)&cudaConv, convHeight * convWidth * sizeof(float)));
        check(cudaMemcpy(cudaConv, convolution.data(), convHeight * convWidth * sizeof(float), cudaMemcpyHostToDevice));
        CudaTiming naiveKernelTiming;
        naiveKernelTiming.Start();

        if (useConstantMemory)
        {
            Naive::NaiveConvolveConstantMemory << < pixelGrid, subGrid >> > (out, input, cudaConv, width, height);
        }
        else {
            Naive::NaiveConvolveNoConstantMemory << < pixelGrid, subGrid >> > (out, input, cudaConv, width, height);
        }
        check(cudaFree(cudaConv));
        naiveKernelTiming.Stop();
        naiveKernelTiming.PrintTime("Kernel Time");
    }
    else {
        CudaTiming basicKernelTiming;
        basicKernelTiming.Start();
        BasicSharedMemory::ConvolveSharedMemory << < pixelGrid, subGrid >> > (out, input, width, height);
        basicKernelTiming.Stop();
        basicKernelTiming.PrintTime("Kernel Time");
    }


    unsigned char* outputPointer = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));
    check(cudaMemcpy(outputPointer, out, pixelCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    check(cudaFree(input));
    check(cudaFree(out));

    return outputPointer;
}


unsigned char* ImageConvolution::ConvolveOptimized(vector<unsigned char>& pixels, vector<float>& convolution, bool useCharSharedMemory)
{
    unsigned char* input;
    unsigned char* out;


    int pixelCount = pixels.size();
    unsigned char* outputPointer = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));

    int pixelsMemory = sizeof(unsigned char) * pixelCount;
    int pixelsMemory_f = sizeof(half) * pixelCount;

    check(cudaMalloc((void**)&input, pixelsMemory));
    check(cudaMemcpy(input, pixels.data(), pixelsMemory, cudaMemcpyHostToDevice));

    vector<float> h_conv_vec;
    vector<float> v_conv_vec;


    float* h_conv;
    float* v_conv;

    bool isSeparable = ConvolutionSeparation::isConvSeparable(convolution, CONV_SIDE_LENGTH, CONV_SIDE_LENGTH, h_conv_vec, v_conv_vec);
    
    if (isSeparable)
    {
        half* pixels_f;
        check(cudaMalloc((void**)&pixels_f, pixelsMemory_f));

        std::cout << "Convolution is separable\n\n";

        //copy horizontal convolution into constant memory
        check(cudaMemcpyToSymbol(constantConv1d, h_conv_vec.data(), CONV_SIDE_LENGTH * sizeof(float)));

        //horizontal convolution
        CudaTiming hTime;
        hTime.Start();
        dim3 horizontalGridDim = dim3((IMAGE_WIDTH + H_BLOCK_X - 1) / H_BLOCK_X, IMAGE_HEIGHT);
        Optimized::ConvolveHorizontalSplitUp << <horizontalGridDim, H_BLOCK_X >> > (pixels_f, input);

        hTime.Stop();
        hTime.PrintTime("Horizontal convolution");

        //copy the vertical convolution into constant memory
        check(cudaMemcpyToSymbol(constantConv1d, v_conv_vec.data(), CONV_SIDE_LENGTH * sizeof(float)));

        dim3 verticalGridDim = dim3((IMAGE_WIDTH + V_BLOCK_X - 1) / V_BLOCK_X, (IMAGE_HEIGHT + V_BLOCK_Y - 1) / V_BLOCK_Y);
        dim3 verticalBlockDim = dim3(V_BLOCK_X, V_BLOCK_Y);

        CudaTiming vTime;
        vTime.Start();
        Optimized::ConvolveVertical <<<verticalGridDim, verticalBlockDim>>> (input, pixels_f);
        vTime.Stop();
        vTime.PrintTime("Vertical convolution");

        check(cudaMemcpy(outputPointer, input, pixelCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        check(cudaFree(pixels_f));
    }
    else
    {
        std::cout << "Convolution is not separable\n\n";
        check(cudaMemcpyToSymbol(constantConv, convolution.data(), convolution.size() * sizeof(float)));

        check(cudaMalloc((void**)&out, pixelsMemory));
        dim3 pixelGrid((IMAGE_WIDTH + BLOCK_X - 1) / BLOCK_X, (IMAGE_HEIGHT + BLOCK_Y - 1) / BLOCK_Y);
        dim3 subGrid(BLOCK_X, BLOCK_Y);

        CudaTiming unseparablekernelTiming;
        unseparablekernelTiming.Start();

        if (useCharSharedMemory)
        {
            Optimized::ConvolveSharedMemoryUnseparableOptimizedChar <<< pixelGrid, subGrid >>> (out, input);
        }
        else {
            Optimized::ConvolveSharedMemoryUnseparableOptimizedInt <<< pixelGrid, subGrid >>> (out, input);
        }
        unseparablekernelTiming.Stop();
        unseparablekernelTiming.PrintTime("Kernel Time");

        
        check(cudaMemcpy(outputPointer, out, pixelCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        check(cudaFree(out));
    }

    

    check(cudaFree(input));

    return outputPointer;
}


unsigned char* ImageConvolution::ConvolveOptimizedPixel4(vector<unsigned char>& pixels, vector<float>& convolution)
{
    pixel4* input;
    pixel* out;


    int pixelCount = pixels.size();
    unsigned char* outputPointer = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));

    int pixelsMemory = sizeof(unsigned char) * pixelCount;

    check(cudaMalloc((void**)&input, pixelsMemory));
    check(cudaMemcpy(input, pixels.data(), pixelsMemory, cudaMemcpyHostToDevice));


    check(cudaMemcpyToSymbol(constantConv, convolution.data(), convolution.size() * sizeof(float)));

    check(cudaMalloc((void**)&out, pixelsMemory));
    dim3 pixelGrid((IMAGE_WIDTH + (4 * BLOCK_X_4_PIXEL) - 1) / (4 * BLOCK_X_4_PIXEL), (IMAGE_HEIGHT + BLOCK_Y_4_PIXEL - 1) / BLOCK_Y_4_PIXEL);
    dim3 subGrid(BLOCK_X, BLOCK_Y);

    CudaTiming unseparablekernelTiming;
    unseparablekernelTiming.Start();

    Experimental::ConvolveSharedMemoryUnseparableOptimizedPixel << < pixelGrid, subGrid >> > (out, input);
    
    unseparablekernelTiming.Stop();
    unseparablekernelTiming.PrintTime("Kernel Time");


    check(cudaMemcpy(outputPointer, out, pixelCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    check(cudaFree(out));
    check(cudaFree(input));

    return outputPointer;
}
