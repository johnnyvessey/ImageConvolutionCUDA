#include <iostream>

#include "lodepng.h"
#include <random>
#include "kernels.cuh"



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
    //convolution[(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH) / 2 ] = 1;

    //convolution[25] = 2.3;
    //vector<float> convolution = { -1,-1,-1,-1,8,-1,-1,-1,-1 };

   

    CudaTiming ct;
    ct.Start();
    vector<unsigned char> newImage = ConvolveImage(pixels, convolution, width, height, CONV_SIDE_LENGTH, CONV_SIDE_LENGTH, false);
    ct.Stop();
    ct.PrintTime("Total function time");

    lodepng::encode("conv_image.png", newImage, width, height);
}