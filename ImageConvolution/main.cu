﻿#include <iostream>
#include "lodepng.h"
#include <random>
#include "kernels.cuh"

#define PI 3.14159265


vector<float> generate_gaussian_blur(int sideLength, float sigma)
{
    vector<float> gaussian_blur_convolution(sideLength * sideLength, 0);

    float sum = 0.0f;
    for (size_t i = 0; i < sideLength; i++)
    {
        for (size_t j = 0; j < sideLength; j++)
        {
            int x = j - sideLength / 2;
            int y = i - sideLength / 2;
            float value = 1.0f / (2.0f * PI * sigma * sigma) * exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            gaussian_blur_convolution[i * sideLength + j] = value;
            sum += value;
        }
    }

    return gaussian_blur_convolution;
}

int main(void) {
    vector<unsigned char> pixels;
    unsigned int width = IMAGE_WIDTH; //required to make variables because lodepng takes width + height as references
    unsigned int height = IMAGE_HEIGHT;
    lodepng::decode(pixels, width, height, "cat_image.png");

    //large box blur
    vector<float> box_blur_convolution(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH, 1.0f / (float)(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH));

    vector<float> gaussian_blur_convolution = generate_gaussian_blur(CONV_SIDE_LENGTH, (float)CONV_SIDE_LENGTH / 6);
    //Add this to make the convolution non-separable
    gaussian_blur_convolution[0] = .2;
   
    CudaTiming ct;
    ct.Start();
    unsigned char* newImage = ImageConvolution::ConvolveOptimized(pixels, gaussian_blur_convolution, IMAGE_WIDTH, IMAGE_HEIGHT, CONV_SIDE_LENGTH, CONV_SIDE_LENGTH);
    ct.Stop();
    ct.PrintTime("Total function time");

    //this is just to make the pictures look better and easier to see the effects of the convolution visually (setting alpha to max value)
    for (size_t i = 3; i < 4 * IMAGE_WIDTH * IMAGE_HEIGHT; i += 4)
    {
        newImage[i] = 255;
    }

    lodepng::encode("conv_image.png", newImage, width, height);
    free(newImage);

}