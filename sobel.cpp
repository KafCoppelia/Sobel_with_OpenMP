#include <stdio.h>
#include <time.h>
#include <string>
#include "sobel.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv)
{
    clock_t start, end;
	double time_taken;

    if (argc != 2) {
        printf("Please input picture directory\n");
        return -1;
    }

    Mat image, gray_img, omp_img;
    image = imread(argv[1], 1);
    if (!image.data) {
        printf("No image data\n");
        return -1;
    }

    cvtColor(image, gray_img, COLOR_RGB2GRAY);
    omp_img = omp_img.zeros(gray_img.size(), gray_img.type());

    start = clock();
    sobel_filter(&gray_img, &omp_img);
    end = clock();
	time_taken = (double)(end - start);
    printf("Time taken is %lf\n", time_taken);

#ifdef DEBUG
    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", image);
    namedWindow("Sobel Image", WINDOW_NORMAL);
    imshow("Sobel Image", omp_img);
    waitKey(0);
#endif

    std::string file(argv[1]);
    std::string extension = file.substr(file.find_last_of("."));
    imwrite(std::string("pics/output") + extension , omp_img);

    return 0;
}