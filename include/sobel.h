#ifndef __SOBEL_H__
#define __SOBEL_H__

#include <opencv2/opencv.hpp>
#include <cmath>
#include <omp.h>

using namespace cv;

int sobel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};
int sobel_y[3][3] = {
	{-1, -2, -1},
	{0,  0,  0},
	{1,  2,  1}
};

void sobel_filter(Mat *img);

void sobel_filter(Mat *img) {
	int gx, gy;
	int i, j;

	int rows = img->rows;
	int cols = img->cols;
	Mat mag = mag.zeros(rows, cols, img->type());

	unsigned char value = 0;

#ifdef USE_OMP
	#pragma omp parallel for privated(i, j, gx, gy, value)
#endif
	for (i = 1; i < rows - 1; i++) {
		for (j = 1; j < cols - 1; j++) {
			gx = (img->data[(i-1)*cols+j-1])*sobel_x[0][0] + (img->data[(i-1)*cols+j])*sobel_x[0][1] + (img->data[(i-1)*cols+j+1])*sobel_x[0][2] + \
				(img->data[i*cols+j-1])*sobel_x[1][0] + (img->data[i*cols+j])*sobel_x[1][1] + (img->data[i*cols+j+1])*sobel_x[1][2]+ \
				(img->data[(i+1)*cols+j-1])*sobel_x[2][0] + (img->data[(i+1)*cols+j])*sobel_x[2][1] + (img->data[(i+1)*cols+j+1])*sobel_x[2][2];
			
			gy = ((img->data[(i-1)*cols+j-1])*sobel_y[0][0] + (img->data[(i-1)*cols+j])*sobel_y[0][1] + (img->data[(i-1)*cols+j+1])*sobel_y[0][2]) + \
				(img->data[i*cols+j-1])*sobel_y[1][0] + (img->data[i*cols+j])*sobel_y[1][1] + (img->data[i*cols+j+1])*sobel_y[1][2]+ \
				(img->data[(i+1)*cols+j-1])*sobel_y[2][0] + (img->data[(i+1)*cols+j])*sobel_y[2][1] + (img->data[(i+1)*cols+j+1])*sobel_y[2][2];
			
			value = (unsigned char) sqrt(gx * gx + gy * gy);
			if (value > 255)
				value = 255;

			std::memcpy(mag.data + (i*cols + j), &value, sizeof(unsigned char));
		}
	}

	std::memcpy(img->data, mag.data, cols*rows*sizeof(unsigned char));
}

#endif /* __SOBEL_H__ */