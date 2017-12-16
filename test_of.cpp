#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <ctime>
#include <string>
#include <unistd.h>
#include <algorithm>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "CycleTimer.h"
#include <time.h>
#include <getopt.h>
//#define DEBUG

#ifdef DEBUG
/* When debugging is enabled, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__); 
#else
/* When debugging is disabled, no code gets generated for these */
#define dbg_printf(...)
#endif

#define SUCCESS 0
#define INVALID_PATCH_SIZE 1
#define OUT_OF_FRAME 2
#define ZERO_DENOMINATOR 3

using namespace cv;
using namespace std;

extern int lkpyramidal_gpu(cv::Mat &I, cv::Mat &J,int levels, int patch_size,
                    vector<Point2f> &ptsI, vector<Point2f> &ptsJ,
                    vector<char> &status);
/* Total images*/
int total_images = 205;
int keypoints = 18;

string format_img = ".jpg";
const int op_pixel_threshold = 10; 


char compute_lk(vector<float> &ix, vector<float> &iy,
                vector<float> &it, pair<float,float> &delta)
{
    float sum_xx = 0.0, sum_yy = 0.0, sum_xt = 0.0,
                        sum_yt = 0.0, sum_xy = 0.0;
    float num_u,num_v, den, u, v;

    /* Calculate sums */
    for (int i = 0; i < ix.size(); i++)
    {
        sum_xx += ix[i] * ix[i];
        sum_yy += iy[i] * iy[i];
        sum_xy += ix[i] * iy[i];
        sum_xt += ix[i] * it[i];
        sum_yt += iy[i] * it[i];
    }

    /* Get numerator and denominator of u and v */
    den = (sum_xx*sum_yy) - (sum_xy * sum_xy);

    if (den == 0.0) return ZERO_DENOMINATOR;

    num_u = (-1.0 * sum_yy * sum_xt) + (sum_xy * sum_yt);
    num_v = (-1.0 * sum_xx * sum_yt) + (sum_xt * sum_xy);

    u = num_u / den;
    v = num_v / den;
    delta.first = u;
    delta.second = v;

    return SUCCESS;
}

void get_vectors(vector< vector<float> > &patch, 
                 vector< vector<float> > &patch_it,
                 int patch_size, vector<float> &ix, vector<float> &iy, 
                 vector<float> &it)
{
    for (int i = 1; i <= patch_size; i++)
        for (int j = 1; j <= patch_size; j++)
        {
            ix.push_back((patch[i][j+1] - patch[i][j-1])/2.0);
            iy.push_back((patch[i+1][j] - patch[i-1][j])/2.0);
        }

    for (int i = 0; i < patch_size; i++)
        for (int j = 0; j < patch_size; j++)
            it.push_back(patch_it[i][j]);

}

char extract_patch(int x, int y, int patch_size,
                   Mat &image, vector< vector<float> > &patch)
{
    int radix = patch_size / 2;

    if ( ((x - radix) < 0) ||
         ((x + radix) >= image.cols) ||
         ((y - radix) < 0) ||
         ((y + radix) >= image.rows))
        return OUT_OF_FRAME;

    for (int i = -radix; i <= radix; i++)
        for (int j = -radix; j <= radix; j++)
            patch[i+radix][j+radix] = image.at<float>(y+i,x+j);

    return SUCCESS;

}

char extract_it_patch(int x_I, int y_I, int x_J, int y_J, Mat &I, Mat &J, 
                      int patch_size, vector< vector<float> > &patch)
{

    int radix = patch_size / 2;

    if (((x_I - radix) < 0) ||
         ((x_I + radix) >= I.cols) ||
         ((y_I - radix) < 0) ||
         ((y_I + radix) >= I.rows))
        return OUT_OF_FRAME;

    if (((x_J - radix) < 0) ||
         ((x_J + radix) >= J.cols) ||
         ((y_J - radix) < 0) ||
         ((y_J + radix) >= J.rows))
        return OUT_OF_FRAME;

    for (int i = -radix; i <= radix; i++)
        for (int j = -radix; j <= radix; j++)
            patch[i+radix][j+radix] = J.at<float>(y_J+i,x_J+j) - 
                                      I.at<float>(y_I+i,x_I+j);

    return SUCCESS;

}
/* Given an OpenCV image 'img', build a gaussian pyramid of size 'levels' */
void build_gaussian_pyramid(Mat &img, int levels, vector<Mat> &pyramid)
{
    pyramid.clear();

    pyramid.push_back(img);

    for(int i = 0; i < levels - 1; i++)
    {
        Mat tmp;
        pyrDown(pyramid[pyramid.size() - 1], tmp);
        pyramid.push_back(tmp);
    }
}



Point2f pyramid_iteration(Point2f ipoint, Point2f jpoint, Mat &I, Mat &J,
                          char &status, int patch_size = 5)
{

    Point2f result;

   /* Extract a patch around the image */
    vector< vector<float> > patch(patch_size + 2,
                                vector<float>(patch_size + 2));
    vector< vector<float> > patch_it(patch_size,
                                vector<float>(patch_size));

    status = extract_patch((int)ipoint.x,(int)ipoint.y,
                           patch_size + 2, I, patch);
    if (status)
        return result;

    status = extract_it_patch(ipoint.x, ipoint.y, jpoint.x, jpoint.y, I, J,
                              patch_size, patch_it);
 
    if (status)
        return result;                         

    /* Get the Ix, Iy and It vectors */
    vector<float> ix, iy, it;
    get_vectors(patch, patch_it, patch_size, ix, iy, it);

    /* Calculate optical flow */
    pair<float,float> delta;
    status = compute_lk(ix, iy, it, delta);
    
    if (status)
        return result;
    
    result.x = jpoint.x + delta.first;
    result.y = jpoint.y + delta.second;

    return result; 
}  

void reescale_cords(vector<Point2f> &coords, float scale)
{
    for (int i = 0; i < coords.size(); i++)
    {
        coords[i].x =  scale * coords[i].x;
        coords[i].y =  scale * coords[i].y;
    }
} 


void reescale_cord(Point2f &coord, float scale)
{
    coord.x =  scale * coord.x;
    coord.y =  scale * coord.y;
}

void run_LKPyramidal(vector<Point2f> &coord_I,
                     vector<Point2f> &coord_J,
                     Mat &prev,
                     Mat &next,
                     vector<char> &status,
                     int levels,
                     int patch_size = 5)
{
    /* Empty coordinates */
    if (coord_I.size() == 0)
        return;

    vector<Point2f> I;
    I.assign(coord_I.begin(), coord_I.end());

    reescale_cords(I,1.0/(float)(1<<(levels-1)));
    
    coord_J.clear();
    coord_J.assign(I.begin(), I.end());

    vector<Mat> I_pyr;
    vector<Mat> J_pyr;

    build_gaussian_pyramid(prev, levels, I_pyr);
    build_gaussian_pyramid(next, levels, J_pyr);

    
    /* Process all pixel requests */
    for (int i = 0; i < coord_I.size(); i++)
    {
        for (int l = levels - 1; l >= 0; l--)
        {
             char status_point = 0;
             Point2f result;

             result = pyramid_iteration(I[i], coord_J[i],I_pyr[l], J_pyr[l],
                              status_point, patch_size);
             if (status_point) {status[i] = status_point; break;}

             coord_J[i] = result;
            
             if (l == 0) break;

             reescale_cord(I[i],2.0);
             reescale_cord(coord_J[i],2.0);
        }
    }

}

void get_opt_flow(vector<Point2f> &coord_in,
                  vector<Point2f> &coord_out,
                  Mat &prev,
                  Mat &next,
                  vector<char> &status,
                  int patch_size = 5)
{
    /* Empty coordinates */
    if (coord_in.size() == 0)
        return;

    Mat It = next - prev;

    /* Even width of patch not valid */
    if (patch_size % 2 == 0)
        status[0] = INVALID_PATCH_SIZE;


    /* Process all pixel requests */
    for (int i = 0; i < coord_in.size(); i++)
    {
        /* Extract a patch around the image */
        vector< vector<float> > patch(patch_size + 2,
                                    vector<float>(patch_size + 2));
        vector< vector<float> > patch_it(patch_size + 2,
                                    vector<float>(patch_size + 2));
  
        status[i] = extract_patch((int)coord_in[i].x,(int)coord_in[i].y,
                      patch_size, prev, patch);


        //if (status[i]) {cout<<"UPSP!"<<endl; continue;} 

        status[i] = extract_patch((int)coord_in[i].x,(int)coord_in[i].y,
                      patch_size, It, patch_it);

        //if (status[i]) {cout<<"UPSP2!"<<endl; continue;} 

        /* Get the Ix, Iy and It vectors */
        vector<float> ix, iy, it;
        get_vectors(patch, patch_it, patch_size, ix, iy, it);

        /* Calculate optical flow */
        pair<float,float> delta;
        status[i] = compute_lk(ix, iy, it, delta);

        //if (status[i]) {cout<<"UPSOF!"<<endl; continue;}

        /* OPTICAL FLOW SUCEED */
        coord_out[i].x =  delta.first + coord_in[i].x;
        coord_out[i].y =  delta.second + coord_in[i].y;
    }

}

void drawKeyPoints(Mat image,vector<int> x, vector<int> y, 
                   std::string output_file){

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for(int i=0;i<x.size();i++){
        if (!x[i] && !y[i]) continue;

        Point center = Point(x[i], y[i]);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    imwrite(output_file, target);
}


void draw_all(Mat image, vector<int> x, vector<int> y, vector<Point2f> &of, 
              vector<Point2f> &custom, std::string output_file)
{

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for(int i=0;i<x.size();i++){
        if (!x[i] && !y[i]) continue;

        Point center = Point(x[i],y[i]);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    for (int i = 0; i < of.size(); i++)
        cv::circle(target,of[i],3,Scalar(0,255,0),1);

    for (int i = 0; i < custom.size(); i++)
        cv::circle(target,custom[i],3,Scalar(0,0,255),1);

    imwrite(output_file, target);
}


void points2crd(vector<int> x, vector<int> y, vector<Point2f> &output_crd){

    for(int i=0;i< x.size() ;i++)
    {
        if (!x[i] && !y[i]) continue;

        Point p = Point2d((float)x[i], (float)y[i]);
        output_crd.push_back(p);
    }

}

void crd2points(vector<float> &x, vector<float> &y, vector<Point2f> &input_crd)
{
    x.clear();
    y.clear();

    for(int i=0;i<input_crd.size();i++)
    {
        x.push_back(input_crd[i].x);
        y.push_back(input_crd[i].y);
    }
}

pair<int,int> find_closest(vector<int> &x_of, vector<int> &y_of, int x, int y)
{
    float min_dist = (float) (INT_MAX);
    int best_x = -1, best_y = -1;

    for (int i = 0; i < x_of.size(); i++)
    {
        float dist = (float) ((x-x_of[i])*(x-x_of[i]) + (y-y_of[i])*(y-y_of[i]));
        
        if (dist < min_dist)
        {        
            min_dist = dist;
            best_x = x_of[i];
            best_y = y_of[i];
        }    
    }
    
    if (min_dist >= op_pixel_threshold)
        return make_pair(-1,-1);
    
    return make_pair(best_x,best_y);
}


void interpolate_next(vector<int> &x_prev, vector<int> &y_prev, 
                      vector<int> &x_of, vector<int> &y_of)
{

    for (int i = 0; i < x_prev.size(); i++)
    {
        if (x_prev[i] == -1)
            continue;

        pair<int,int> new_cord = find_closest(x_of,y_of,x_prev[i],y_prev[i]);
        x_prev[i] = new_cord.first;
        y_prev[i] = new_cord.second;
    }
}

void get_statistics(vector<int> &x_itp, vector<int> &y_itp,
                    vector<float> &x_of, vector<float> &y_of,
                    vector<float> &maxim, vector<float> &mean,
                    vector<float> &minim, vector<float> &median)
{

    vector<float> distances;
    float dist = 0.0, total = 0.0;
    int n = x_itp.size();

    for (int i = 0; i < x_itp.size(); i++)
    {
        if (x_itp[i] == -1)
            continue;
        
        dist = (float) (((float)x_itp[i]-x_of[i])*((float)x_itp[i]-x_of[i]) + 
                ((float)y_itp[i]-y_of[i])*((float)y_itp[i]-y_of[i]));        
        dist = sqrt(dist);
        distances.push_back(dist);         
    }
    if (!distances.size()) return;

    std::sort(distances.begin(),distances.end());
    int nk = distances.size(); 
    for (int k = 0; k < nk; k++) total += distances[k];

    maxim.push_back(distances.back());
    minim.push_back(distances[0]);
    mean.push_back(total/(float)nk);

    if (n % 2 == 0) 
        median.push_back((distances[nk/2] + distances[nk/2 -1]) / 2.0);
    else median.push_back(distances[nk/2]);
     
}

int main(int argc, char ** argv) 
{

    int first_frame = -1, last_frame = -1, option = 0;
    char *in_frames_path = NULL, *out_frames_path = NULL,
         *stats_file = NULL, *cords_file = NULL;
    bool verbose = false;

    /* Get input arguments */ 
    while ((option = getopt(argc, argv,"s:e:p:c:f:o:v")) != -1) 
    {
        switch (option) {
             case 's' : first_frame = atoi(optarg);
                 break;
             case 'e' : last_frame = atoi(optarg);
                 break;
             case 'p' : in_frames_path = optarg; 
                 break;
             case 'c' : cords_file = optarg;
                 break;
             case 'f' : stats_file = optarg;
                 break;
             case 'o' : out_frames_path = optarg;
                 break;
             case 'v': verbose = true;
                 break;
             default:  
                 exit(EXIT_FAILURE);
        }
    }

    /* Check mandatory parameters */
    if (first_frame == -1) {cout<<"Invalid first frame"<<endl; return 0;}
    if (last_frame == -1) {cout<<"Invalid last frame"<<endl; return 0;}
    if (!in_frames_path) {cout<<"Specify input frames path!"<<endl; return 0;}
    if (!cords_file) {cout<<"Specify coordinate log!"<<endl; return 0;}

    /* Input frames in CV_8U and float formats */
    Mat input, input_float;
    /* Current and previous frames for tracking. CV_8U and float */
    Mat prev, current, fprev, fcurrent;
    /* Tracking points: OpenCV and custom. Index 0: previous frame
     * Index 1: current frame
     */       
    vector<Point2f> opencv_points[2];
    vector<Point2f> custom_points[2];
    /* String version of paths */
    string out_frames_path_str = "", in_frames_path_str = "";
        
    in_frames_path_str.assign(in_frames_path);
    
    if (out_frames_path)
        out_frames_path_str.assign(out_frames_path);
    

    /* Redirect input from coordinates file */
    freopen(cords_file,"r",stdin);

    /* OpenCV tracker parameters */
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(21,21);

    /* OpenPose x,y keypoints: 'ground_truth' coordinates */
    vector<int> x_truth, y_truth;
    vector<float> minim, maxim, mean, median;
    /* Estimated OpenPose points in the current frame with respect
     * to the original tracking frame
     */  
    vector<int> x_itp, y_itp;
    vector<float> x_of, y_of;
    /* Status vectors for Optical Flow */
    vector<char> custom_status;
    vector<uchar> opencv_status;

    float fx, fy, conf;

    for (int i = 0; i < last_frame + 1; i++)
    {
        int persons;
        
        if (verbose && i >= first_frame)
            cout<<"Processing Frame "<<i<<endl;

        string number = ""; 
        char buffer[15];
        sprintf(buffer,"%d",i);
        string temp(buffer);
 
        if (i > 99) number = temp;
        else if (i > 9) number = "0" + temp;
        else number = "00" + temp;
        
        /* Load image */
        string img = in_frames_path_str + number + format_img; 
        string output_image = out_frames_path_str + number + format_img;        
        /* Get current frame in both fromats (CV_8U and float) */
        input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);                                      
        input.convertTo(input_float, CV_32F);

        cin>>persons;
        
	x_truth.clear();
        y_truth.clear();

        for (int p = 0; p < persons; p++)
        {
	    /* Add all keypoints */
	    for (int k = 0; k < keypoints; k++)
	    {	      	    
	        cin>>fx>>fy>>conf;
                /* Keypoints with less than 1% confidence are discarded */
                if (conf < 0.01)
	            continue;	

                x_truth.push_back((int)fx);
                y_truth.push_back((int)fy);
            }
	}

        /* Set tracking points */
        if (i == first_frame)
        {
            fcurrent = input_float.clone();
            current = input;
            /* Set initial optical flow points to OpenPose first 
             * tracking frame
             */      
            points2crd(x_truth, y_truth, opencv_points[1]);
            points2crd(x_truth, y_truth, custom_points[1]);

            custom_points[0].resize(custom_points[1].size());

            x_itp.assign(x_truth.begin(), x_truth.end());
            y_itp.assign(y_truth.begin(), y_truth.end());
            x_of.assign(x_truth.begin(), x_truth.end());
            y_of.assign(y_truth.begin(), y_truth.end());

            custom_status.assign(custom_points[1].size(),0);
            opencv_status.assign(custom_points[1].size(),0); 
        } 
        /* Draw OF keypoint estimates for frames (s,e] */
        if (i > first_frame && i < last_frame + 1)
        {
            vector<float> err;

            /* Swap previous and current points and frames */
            std::swap(opencv_points[1], opencv_points[0]);   
            std::swap(custom_points[1], custom_points[0]);
            cv::swap(fcurrent, fprev);
            cv::swap(current, prev);

            fcurrent = input_float.clone();
            current = input;
            /* Estimate corresponding OpenPose points on the current
             * frame with respect to the original frame. Used only for 
             * statistics
             */       
            interpolate_next(x_itp, y_itp, x_truth, y_truth);

            calcOpticalFlowPyrLK(prev, current, opencv_points[0], 
                                 opencv_points[1], 
                                 opencv_status, err, winSize, 5, 
                                 termcrit, 0, 0.001);
            
            if (stats_file)
            {
                crd2points(x_of, y_of, opencv_points[1]);                         
                get_statistics(x_itp, y_itp, x_of, y_of, 
                               maxim, mean,minim,median);
                
                if (verbose)
                   cout<<"MAXIMUM: "<<maxim.back()<<",  MINIMUM: "<<minim.back()
                       <<", MEAN: "<<mean.back()<<", MEDIAN: "
                       <<median.back()<<endl; 
            }
            
	    lkpyramidal_gpu(fprev, fcurrent, 5, 21, custom_points[0],    
                            custom_points[1], custom_status);

            //run_LKPyramidal(custom_points[0], custom_points[1], 
            //fprev, fcurrent, mystatus,3, 21);
            
           if (out_frames_path)
            draw_all(input_float,x_truth,y_truth, opencv_points[1], 
                     custom_points[1], output_image);
        }     
        else 
            if (out_frames_path)
                drawKeyPoints(input_float, x_truth, y_truth,output_image);
    }
    
    return 0;

}
