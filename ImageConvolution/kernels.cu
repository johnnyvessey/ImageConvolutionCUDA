#include "kernels.cuh"

__constant__ float constantConv[CONV_SIDE_LENGTH * CONV_SIDE_LENGTH];

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

__global__ void NaiveConvolve(unsigned char* out, unsigned char* pixels, float* cudaConv, int width, int height)
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

vector<unsigned char> ConvolveImage(vector<unsigned char>& pixels, vector<float>& convolution, int width, int height, int convHeight, int convWidth, bool naive)
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


    CudaTiming kernelTiming;
    kernelTiming.Start();
    if (naive)
    {
        float* cudaConv;
        check(cudaMalloc((void**)&cudaConv, convHeight * convWidth * sizeof(float)));
        check(cudaMemcpy(cudaConv, convolution.data(), convHeight * convWidth * sizeof(float), cudaMemcpyHostToDevice));
        NaiveConvolve <<< pixelGrid, subGrid >>> (out, input, cudaConv, width, height);
        check(cudaFree(cudaConv));

    }
    else {
        ConvolveSharedMemory <<< pixelGrid, subGrid >>> (out, input, width, height);
    }
    //ConvolveSharedMemoryNoZ << < dim3(width / BLOCK_X,height / BLOCK_Y), dim3(BLOCK_X, BLOCK_Y) >> > (out, input, width, height);
    //ConvolveSharedMemory1D <<< (width * height) / 1024, dim3(1024, 4) >>> (out, input, width, height);
    //ConvolveSharedMemoryAlign<<<dim3(1,width / BLOCK_Y, height / BLOCK_Z), dim3(BLOCK_X, BLOCK_Y, BLOCK_Z) >> > (out, input, width, height);
    kernelTiming.Stop();
    kernelTiming.PrintTime("Kernel Time");

    //std::cout << "Avg time: " << totalTime / (float)num_iter << " ms\n";


    unsigned char* outputPointer = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));
    check(cudaMemcpy(outputPointer, out, pixelCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    check(cudaFree(input));

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
