#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <time.h>
#include <string>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace std;
using namespace cv;

void sobel_cpu(Mat* orig, Mat* cpu);
void sobel_omp(Mat* orig, Mat* cpu);
__global__ void sobel_cuda(unsigned char *orig, unsigned char *gpu, int imgHeight, int imgWidth);

void sobel_cpu(Mat* orig, Mat* cpu) {
    int x, y;
    int width = orig->cols;
	int height = orig->rows;
    float gx, gy;

    for (y = 1; y < height-1; y++) {
        for (x = 1; x < width-1; x++) {
            gx = (-1*orig->data[(y-1)*width + (x-1)]) + (-2*orig->data[y*width+(x-1)]) + (-1*orig->data[(y+1)*width+(x-1)]) +
                (orig->data[(y-1)*width + (x+1)]) + (2*orig->data[y*width+(x+1)]) + (orig->data[(y+1)*width+(x+1)]);
            
            gy = (orig->data[(y-1)*width + (x-1)]) + (2*orig->data[(y-1)*width+x]) + (orig->data[(y-1)*width+(x+1)]) +
                (-1*orig->data[(y+1)*width + (x-1)]) + (-2*orig->data[(y+1)*width+x]) + (-1*orig->data[(y+1)*width+(x+1)]);
            
            cpu->data[y*width + x] = sqrt(gx*gx + gy*gy) > 255 ? 255 : sqrt(gx*gx + gy*gy);
        }
    }
}

void sobel_omp(Mat* orig, Mat* cpu) {
    int x, y;
    int width = orig->cols;
	int height = orig->rows;
    float gx, gy;

    #pragma omp parallel for private(y), private(x), private(gx), private(gy)
    for (y = 1; y < height-1; y++) {
        for (x = 1; x < width-1; x++) {
            gx = (-1*orig->data[(y-1)*width + (x-1)]) + (-2*orig->data[y*width+(x-1)]) + (-1*orig->data[(y+1)*width+(x-1)]) +
                (orig->data[(y-1)*width + (x+1)]) + (2*orig->data[y*width+(x+1)]) + (orig->data[(y+1)*width+(x+1)]);
            
            gy = (orig->data[(y-1)*width + (x-1)]) + (2*orig->data[(y-1)*width+x]) + (orig->data[(y-1)*width+(x+1)]) +
                (-1*orig->data[(y+1)*width + (x-1)]) + (-2*orig->data[(y+1)*width+x]) + (-1*orig->data[(y+1)*width+(x+1)]);
            
            cpu->data[y*width + x] = sqrt(gx*gx + gy*gy) > 255 ? 255 : sqrt(gx*gx + gy*gy);
        }
    }
}

__global__ void sobel_cuda(unsigned char *orig, unsigned char *gpu, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    float gx, gy;
 
    if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
    {
        gx = orig[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * orig[yIndex * imgWidth + xIndex + 1] + orig[(yIndex + 1) * imgWidth + xIndex + 1]
            - (orig[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * orig[yIndex * imgWidth + xIndex - 1] + orig[(yIndex + 1) * imgWidth + xIndex - 1]);
        
        gy = orig[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * orig[(yIndex - 1) * imgWidth + xIndex] + orig[(yIndex - 1) * imgWidth + xIndex + 1]
            - (orig[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * orig[(yIndex + 1) * imgWidth + xIndex] + orig[(yIndex + 1) * imgWidth + xIndex + 1]);
        
        gpu[yIndex * imgWidth + xIndex] = sqrt(gx*gx + gy*gy) > 255 ? 255 : sqrt(gx*gx + gy*gy);
    }
}
 
int main(int argc, char** argv)
{
    clock_t cpu_start, cpu_end, omp_start, omp_end;
	double time_taken;

    if (argc != 2) {
        printf("Please input picture directory\n");
        return -1;
    }

    Mat image, orig_img, cpu_img, omp_img, gpu_img;
    image = imread(argv[1], 1);
    if (!image.data) {
        printf("No image data\n");
        return -1;
    }

    cvtColor(image, orig_img, COLOR_RGB2GRAY);
    cpu_img = cpu_img.zeros(orig_img.size(), orig_img.type());
    omp_img = omp_img.zeros(orig_img.size(), orig_img.type());
    gpu_img = gpu_img.zeros(orig_img.size(), orig_img.type());
    
    /* Start CPU test */
    cpu_start = clock();
    sobel_cpu(&orig_img, &cpu_img);
    cpu_end = clock();
    /* End CPU test */

    /* Start OMP test */
    omp_start = clock();
    sobel_omp(&orig_img, &omp_img);
    omp_end = clock();
    /* End OMP test */
    
    int imgHeight = orig_img.rows;
    int imgWidth = orig_img.cols;
 
    //创建GPU内存
    unsigned char *d_in, *d_out;

    cudaMalloc((void**)&d_in, imgWidth * imgHeight * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgWidth * imgHeight * sizeof(unsigned char));
 
    // CPU->GPU
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaMemcpy(d_in, orig_img.data, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, imgWidth * imgHeight * sizeof(unsigned char));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); //避免gpu线程没有完全执行结束
    
    // 调用核函数
    sobel_cuda <<< blocksPerGrid, threadsPerBlock >>> (d_in, d_out, imgHeight, imgWidth);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaError_t cudaerror = cudaDeviceSynchronize(); 
    if (cudaerror != cudaSuccess) {
        printf("CUDA error: %d\n", cudaerror);
    }
 
    // GPU->CPU
    cudaMemcpy(gpu_img.data, d_out, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    /* CPU time */
	time_taken = (double)(cpu_end - cpu_start);
    printf("CPU time taken is %lfs\n", time_taken/CLOCKS_PER_SEC);

    /* OMP time */
    time_taken = (double)(omp_end - omp_start);
    printf("OMP time taken is %lfs\n", time_taken/CLOCKS_PER_SEC);

    /* GPU time */
    printf("GPU time taken is %fs\n", elapsedTime);

    std::string file(argv[1]);
    std::string extension = file.substr(file.find_last_of("."));
    imwrite(std::string("pics/cpu_output") + extension , cpu_img);
    imwrite(std::string("pics/omp_output") + extension , omp_img);
    imwrite(std::string("pics/gpu_output") + extension , gpu_img);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
 
    return 0;
}