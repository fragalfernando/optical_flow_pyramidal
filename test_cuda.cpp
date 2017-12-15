#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"

using namespace cv;
using namespace std;


extern void call_bkernel(cv::cuda::GpuMat img, int w, int h);
extern int lkpyramidal_gpu(cv::Mat &I, cv::Mat &J,int levels, int patch_size,
                    vector<Point2f> &ptsI, vector<Point2f> &ptsJ,
                    vector<char> &status);

int main (int argc, char* argv[])
{
    cv::Mat f1 = cv::imread("./data/065.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat f2 = cv::imread("./data/066.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat frame1, frame2;
    f1.convertTo(frame1, CV_32F);
    f2.convertTo(frame2, CV_32F);

    vector<Point2f> ptsI;
    
    Point2f t;
    t.x = 400;
    t.y = 256;    
    ptsI.push_back(t);

    t.x = 410;
    t.y = 256; 
    ptsI.push_back(t);

    t.x = 398;
    t.y = 313; 
    ptsI.push_back(t);

    t.x = 410;
    t.y = 307; 
    ptsI.push_back(t);

    vector<Point2f> ptsJ;

    vector<char> status(4,0);

    lkpyramidal_gpu(frame1, frame2, 3, 25, ptsI, ptsJ, status);

    for (int i = 0; i < ptsJ.size(); i++)
        cout<< ptsJ[i].x<<"  "<<ptsJ[i].y<<endl;
    //cv::imshow("Result",src_host);
    //cv::waitKey();   

}
