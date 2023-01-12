#include <iostream>
#include "lodepng.h"
#include <random>
#include "kernels.cuh"

#define PI 3.14159265

enum KernelType
{
    NAIVE_NO_CONSTANT_MEMORY,
    NAIVE_CONSTANT_MEMORY,
    BASIC_SHARED_MEMORY,
    SHARED_MEMORY_OPT_CHAR,
    SHARED_MEMORY_OPT_INT,
    EXPERIMENT_4_PIXEL_BLOCK
};
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

    for (size_t i = 0; i < sideLength * sideLength; i++)
    {
        gaussian_blur_convolution[i] /= sum;
    }
    return gaussian_blur_convolution;
}

//make sure convolution.size() is the same as CONV_SIDE_LENGTH * CONV_SIDE_LENGTH
void CreateConvolvedImage(vector<float> convolution, std::string inputFilename, std::string filename, KernelType type)
{
    unsigned int width = IMAGE_WIDTH; //required to make variables because lodepng takes width + height as references
    unsigned int height = IMAGE_HEIGHT;
    vector<unsigned char> pixels;

    lodepng::decode(pixels, width, height, inputFilename);


    CudaTiming ct;
    ct.Start();
    
    unsigned char* newImage;
    switch (type)
    {
        case NAIVE_NO_CONSTANT_MEMORY: newImage = ImageConvolution::ConvolveImage(pixels, convolution, IMAGE_WIDTH, IMAGE_HEIGHT, CONV_SIDE_LENGTH, CONV_SIDE_LENGTH, true, false); break;
        case NAIVE_CONSTANT_MEMORY: newImage = ImageConvolution::ConvolveImage(pixels, convolution, IMAGE_WIDTH, IMAGE_HEIGHT, CONV_SIDE_LENGTH, CONV_SIDE_LENGTH, true, true); break;
        case BASIC_SHARED_MEMORY: newImage = ImageConvolution::ConvolveImage(pixels, convolution, IMAGE_WIDTH, IMAGE_HEIGHT, CONV_SIDE_LENGTH, CONV_SIDE_LENGTH, false, true); break;
        case SHARED_MEMORY_OPT_CHAR: newImage = ImageConvolution::ConvolveOptimized(pixels, convolution, true); break;
        case SHARED_MEMORY_OPT_INT:  newImage = ImageConvolution::ConvolveOptimized(pixels, convolution, false); break;
        case EXPERIMENT_4_PIXEL_BLOCK: newImage = ImageConvolution::ConvolveOptimizedPixel4(pixels, convolution); break;
    }
    ct.Stop();
    ct.PrintTime("Total function time");

    //this is just to make the pictures look better and easier to see the effects of the convolution visually (setting alpha to max value)
    for (size_t i = 3; i < 4 * IMAGE_WIDTH * IMAGE_HEIGHT; i += 4)
    {
        newImage[i] = 255;
    }

    lodepng::encode(filename, newImage, width, height);
    free(newImage);

}

void EdgeDetection(std::string inputFilename, std::string filename)
{
    
    unsigned int width = IMAGE_WIDTH; //required to make variables because lodepng takes width + height as references
    unsigned int height = IMAGE_HEIGHT;
    vector<unsigned char> pixels;

    lodepng::decode(pixels, width, height, inputFilename);

    //blur image slighly before applying sobel filters
    vector<float> blur = generate_gaussian_blur(CONV_SIDE_LENGTH, 2);
    unsigned char* blurImage = ImageConvolution::ConvolveOptimized(pixels, blur, false);
    for (size_t i = 0; i < pixels.size(); i++)
    {     
        pixels[i] = blurImage[i];
    }
    free(blurImage);

    vector<float> conv1 = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    vector<float> conv2 = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    unsigned char* firstPass = ImageConvolution::ConvolveOptimized(pixels, conv1, false);

    for (size_t i = 0; i < pixels.size(); i++)
    {
        pixels[i] = firstPass[i];
    }
    free(firstPass);

    unsigned char* secondPass = ImageConvolution::ConvolveOptimized(pixels, conv2, false);
    //this is just to make the pictures look better and easier to see the effects of the convolution visually (setting alpha to max value)
    for (size_t i = 3; i < 4 * IMAGE_WIDTH * IMAGE_HEIGHT; i += 4)
    {
        secondPass[i] = 255;
    }

    lodepng::encode(filename, secondPass, width, height);
    free(secondPass);


}


int main(void) {

    //large box blur
    vector<float> box_blur_convolution(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH, 1.0f / (float)(CONV_SIDE_LENGTH * CONV_SIDE_LENGTH));

    vector<float> gaussian_blur_convolution = generate_gaussian_blur(CONV_SIDE_LENGTH, (float)CONV_SIDE_LENGTH / 4);

    std::cout << "Separable Optimized Convolution:\n";
    CreateConvolvedImage(gaussian_blur_convolution, "cat_image_large.png", "conv_image_separable.png", SHARED_MEMORY_OPT_CHAR);


    //Add this to make the convolution non-separable
    gaussian_blur_convolution[0] = .2;

    std::cout << "\n------------------------\n\nNon-separable Optimized Convolution Int Shared Memory:\n";
    CreateConvolvedImage(gaussian_blur_convolution, "cat_image_large.png", "conv_image_unseparable.png", SHARED_MEMORY_OPT_INT);

    std::cout << "\n------------------------\n\nNon-separable Optimized Convolution Char Shared Memory:\n";
    CreateConvolvedImage(gaussian_blur_convolution, "cat_image_large.png", "conv_image_unseparable.png", SHARED_MEMORY_OPT_CHAR);

    std::cout << "\n------------------------\n\nBasic Shared Memory Convolution:\n";
    CreateConvolvedImage(gaussian_blur_convolution, "cat_image_large.png", "conv_image_basic_sm.png", BASIC_SHARED_MEMORY);

    std::cout << "\n------------------------\n\nNaive Convolution with Constant Memory:\n";
    CreateConvolvedImage(gaussian_blur_convolution, "cat_image_large.png", "conv_image_naive_cm.png", NAIVE_CONSTANT_MEMORY);

    std::cout << "\n------------------------\n\nNaive Convolution with No Constant Memory:\n";
    CreateConvolvedImage(gaussian_blur_convolution, "cat_image_large.png", "conv_image_naive_no_cm.png", NAIVE_NO_CONSTANT_MEMORY);

    std::cout << "\n------------------------\n\Experimental 4-pixel block Convolution:\n";
    CreateConvolvedImage(gaussian_blur_convolution, "cat_image_large.png", "conv_image_experimental.png", EXPERIMENT_4_PIXEL_BLOCK);

    //EdgeDetection("cat_image.png", "sobel_image.png");

}