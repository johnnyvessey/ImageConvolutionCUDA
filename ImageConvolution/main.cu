#include <iostream>
#include "lodepng.h"
#include <random>
#include "kernels.cuh"

#define PI 3.14159265


int main(void) {
    vector<unsigned char> pixels;
    unsigned int width = IMAGE_WIDTH; //required to make variables because lodepng takes width + height as references
    unsigned int height = IMAGE_HEIGHT;
    lodepng::decode(pixels, width, height, "cat_image.png");

    //large box blur
    vector<float> box_blur_convolution(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH, 1.0f / (float)(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH));
    vector<float> gaussian_blur_convolution(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH, 0);
    float sigma = (float)CONV_SIDE_LENGTH / 6.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < CONV_SIDE_LENGTH; i++)
    {
        for (size_t j = 0; j < CONV_SIDE_LENGTH; j++)
        {
            int x = j - CONV_SIDE_LENGTH / 2;
            int y = i - CONV_SIDE_LENGTH / 2;
             float value =  1.0f / (2.0f * PI * sigma * sigma) * exp(-(x * x + y * y) / (2.0f * sigma * sigma));
             gaussian_blur_convolution[i * CONV_SIDE_LENGTH + j] = value;
             sum += value;
        }
    }

    //Add this to make the convolution non-separable
    gaussian_blur_convolution[0] = .2;
   
    //vector<float> sobel_convolution_step_1 = { -1,0 ,1,-2,0,2,-1,0,1 };
    CudaTiming ct;
    ct.Start();
    unsigned char* newImage = ImageConvolution::ConvolveOptimized(pixels, gaussian_blur_convolution, IMAGE_WIDTH, IMAGE_HEIGHT, CONV_SIDE_LENGTH, CONV_SIDE_LENGTH);
    ct.Stop();
    ct.PrintTime("Total function time");

    //this is just to make the pictures look better and easier to see the effects of the convolution visually
    for (size_t i = 3; i < 4 * IMAGE_WIDTH * IMAGE_HEIGHT; i += 4)
    {
        newImage[i] = 255;
    }

    lodepng::encode("conv_image.png", newImage, width, height);
    free(newImage);

}