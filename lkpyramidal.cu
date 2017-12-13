#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <ctime>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"


__global__ void bkernel (float* img,
                         int w,
                         int h)
{
    /*2D Index of current thread */
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    img[(y*w) + x]*=1.80;
   
    if (img[(y*w) + x] > 255.0)
        img[(y*w) + x] = 255.0;

}

void call_bkernel(cv::cuda::GpuMat img, int w, int h)
{
    /* Device image */
    float *img_ptr = (float*) img.ptr<float>();

    /* BLock width */
    int block_width = 16;


    /* Calculate Grid Size */
    const dim3 block(block_width, block_width);
    const dim3 grid( (w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    /* Launch Kernel */
    bkernel<<<grid,block>>>(img_ptr, w, h);
    cudaDeviceSynchronize();

    printf("Finished calling kernel\n");

}
