#include "common.h"
#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;
  
__global__ void addKernel(uchar3 **pSrcImg,  uchar3* pDstImg, int imgW, int imgH)  
{  
    int x = threadIdx.x + blockDim.x * blockIdx.x;  
    int y = threadIdx.y + blockDim.y * blockIdx.y;  
    if (x < imgW && y < imgH)
    {
        
        int offset = y * imgW + x;  
        
        uchar3 pixel1 = pSrcImg[0][offset];
        uchar3 pixel2 = pSrcImg[1][offset];
        pDstImg[offset].x =uchar(pixel1.x + pixel2.x);
        pDstImg[offset].y =uchar(pixel1.y + pixel2.y);
        pDstImg[offset].z =uchar(pixel1.z + pixel2.z);
    }
}  

int main()  
{  
   
    Mat img[2];
    img[0]=imread("../data/test.jpg");
    img[1]=imread("../data/NASA_Mars_Rover.jpg");
    int imgH=img[0].rows;
    int imgW=img[0].cols;
    
    Mat dstImg=Mat::zeros(imgH, imgW, CV_8UC3);
    
    uchar3 **pImg=(uchar3**)malloc(sizeof(uchar3*)*2); 

    
    uchar3 **pDevice;
    uchar3 *pDeviceData;
    uchar3 *pDstImgData;

    
    
    cudaMalloc(&pDstImgData, imgW*imgH*sizeof(uchar3));
    
    cudaMalloc(&pDevice, sizeof(uchar3*)*2);
    
    cudaMalloc(&pDeviceData, sizeof(uchar3)*imgH*imgW*2);
    
    
    for (int i=0; i<2; i++)
    {
        pImg[i]=pDeviceData+i*imgW*imgH;
    }

    
    cudaMemcpy(pDevice, pImg, sizeof(uchar3*)*2, cudaMemcpyHostToDevice);
   
    cudaMemcpy(pDeviceData, img[0].data, sizeof(uchar3)*imgH*imgW, cudaMemcpyHostToDevice);
    cudaMemcpy(pDeviceData+imgH*imgW, img[1].data, sizeof(uchar3)*imgH*imgW, cudaMemcpyHostToDevice);

   
    dim3 block(8, 8);
    dim3 grid( (imgW+block.x-1)/block.x, (imgH+block.y-1)/block.y);
    addKernel<<<grid, block>>>(pDevice, pDstImgData, imgW, imgH);
    cudaThreadSynchronize();

    
    cudaMemcpy(dstImg.data, pDstImgData, imgW*imgH*sizeof(uchar3), cudaMemcpyDeviceToHost);
    imwrite("../Thsis.jpg", dstImg);
    CHECK(cudaDeviceReset());
    return 0;
}  