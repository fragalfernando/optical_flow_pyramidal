#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"

using namespace cv;
using namespace std;


extern void call_bkernel(cv::cuda::GpuMat img, int w, int h);
/* Given an OpenCV image 'img', build a gaussian pyramid of size 'levels' */
void build_gaussian_pyramid_gpu(Mat &img, int levels, vector<cv::cuda::GpuMat> &pyramid)
{
    cv::cuda::GpuMat current;
    pyramid.clear();

    current.upload(img);
    pyramid.push_back(current);

    for(int i = 0; i < levels - 1; i++)
    {
        cv::cuda::GpuMat tmp;
        cv::cuda::pyrDown(pyramid[pyramid.size() - 1], tmp);
        pyramid.push_back(tmp);
    }
}

int main (int argc, char* argv[])
{
    cv::Mat src_host = cv::imread("./data/065.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat src_float;
    src_host.convertTo(src_float, CV_32F);

    vector<cv::cuda::GpuMat> pyramid;
    build_gaussian_pyramid_gpu(src_float,3,pyramid);

    //cv::cuda::GpuMat dst, src, src2;
    //src.upload(src_host);
    //cv::cuda::pyrDown(src,src2);
    //cv::cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
    //cv::cuda::pyrDown(src2,dst);
    call_bkernel(pyramid[0], pyramid[0].cols, pyramid[0].rows);
    cv::Mat result_host;

    pyramid[0].download(result_host);

    result_host.convertTo(src_host, CV_8U);

    cv::imshow("Result",src_host);
    cv::waitKey();   

}
