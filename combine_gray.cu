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
  
__global__ void addKernel(uchar **pSrcImg,  uchar* pDstImg, int imgW, int imgH)  
{  
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;  
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;  
    if (tidx<imgW && tidy<imgH)
    {
        int idx=tidy*imgW+tidx;
        uchar lenaValue=pSrcImg[0][idx];
        uchar moonValue=pSrcImg[1][idx];
        pDstImg[idx]= uchar(0.5*lenaValue+0.5*moonValue);
    }
}  

int main()  
{  
    
    Mat img[2];
    img[0]=imread("../data/222.jpg", 0);
    img[1]=imread("../data/333.jpg", 0);
    int imgH=img[0].rows;
    int imgW=img[0].cols;
    
    Mat dstImg=Mat::zeros(imgH, imgW, CV_8UC1);
    
    uchar **pImg=(uchar**)malloc(sizeof(uchar*)*2); 

    
    uchar **pDevice;
    uchar *pDeviceData;
    uchar *pDstImgData;

  
    
    cudaMalloc(&pDstImgData, imgW*imgH*sizeof(uchar));
    
    cudaMalloc(&pDevice, sizeof(uchar*)*2);
    
    cudaMalloc(&pDeviceData, sizeof(uchar)*imgH*imgW*2);
    
    
    for (int i=0; i<2; i++)
    {
        pImg[i]=pDeviceData+i*imgW*imgH;
    }

    
    
    cudaMemcpy(pDevice, pImg, sizeof(uchar*)*2, cudaMemcpyHostToDevice);
    
    cudaMemcpy(pDeviceData, img[0].data, sizeof(uchar)*imgH*imgW, cudaMemcpyHostToDevice);
    cudaMemcpy(pDeviceData+imgH*imgW, img[1].data, sizeof(uchar)*imgH*imgW, cudaMemcpyHostToDevice);

   
    dim3 block(8, 8);
    dim3 grid( (imgW+block.x-1)/block.x, (imgH+block.y-1)/block.y);
    addKernel<<<grid, block>>>(pDevice, pDstImgData, imgW, imgH);
    cudaThreadSynchronize();

   
    cudaMemcpy(dstImg.data, pDstImgData, imgW*imgH*sizeof(uchar), cudaMemcpyDeviceToHost);
    imwrite("../Thsis.jpg", dstImg);
    CHECK(cudaDeviceReset());
    return 0;
}  