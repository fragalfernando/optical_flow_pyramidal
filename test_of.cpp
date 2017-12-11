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
/* When debugging is disnabled, no code gets generated for these */
#define dbg_printf(...)
#endif

#define SUCCESS 0
#define INVALID_PATCH_SIZE 1
#define OUT_OF_FRAME 2
#define ZERO_DENOMINATOR 3

using namespace cv;
using namespace std;

/* Total images*/
int total_images = 205;
int keypoints = 18;

/* Frame dimensions and scale alpha */
int frame_height = 0;
int frame_width = 0;
float alpha = 40.0;

/* Image and coordinates directory */
string test_dir = "./data/";
string output_dir = "./output/";
string format_img = ".jpg";
string coordinates_fname = "output.log";

const char *coordinate_file = "./data/output.log";
const int op_pixel_threshold = 10; 

/* Pixel percentiles */
const float pixel_percentiles[] = {3,6,9,12,99999};
const int percentiles_length = 5;


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

void get_vectors(vector< vector<float> > &patch, vector< vector<float> > &patch_it,
                 int patch_size, vector<float> &ix, vector<float> &iy, vector<float> &it)
{
    for (int i = 1; i <= patch_size; i++)
        for (int j = 1; j <= patch_size; j++)
        {
            it.push_back(patch_it[i][j]);
            ix.push_back((patch[i][j+1] - patch[i][j-1])/2.0);
            iy.push_back((patch[i+1][j] - patch[i-1][j])/2.0);
        }
}

char extract_patch(int x, int y, int patch_size,
                   Mat &image, vector< vector<float> > &patch)
{
    int radix = patch_size / 2;

    if ( ((x - (radix + 1)) < 0) ||
         ((x + (radix + 1)) >= image.cols) ||
         ((y - (radix + 1)) < 0) ||
         ((y + (radix + 1)) >= image.rows))
        return OUT_OF_FRAME;

    for (int i = -radix-1; i <= radix+1; i++)
        for (int j = -radix-1; j <= radix+1; j++)
            patch[i+radix+1][j+radix+1] = image.at<float>(y+i,x+j);

    return SUCCESS;

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

void drawKeyPoints(Mat image,vector<int> x, vector<int> y,std::string output_file){

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for(int i=0; i< x.size() ; i++)
    {
        if (!x[i] && !y[i]) continue;

        Point center = Point(x[i], y[i]);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    imwrite(output_file, target);
}

void draw_both(Mat image, vector<int>x, vector<int> y,vector<Point2f> &of, std::string output_file){

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for(int i=0;i<x.size();i++)
    {
        if (!x[i] && !y[i]) continue;

        Point center = Point(x[i], y[i]);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    for (int i = 0; i < of.size(); i++)
        cv::circle(target,of[i],3,Scalar(0,255,0),1);

    imwrite(output_file, target);
}

void draw_all(Mat image, vector<int> x, vector<int> y, vector<Point2f> &of, vector<Point2f> &custom, 
              std::string output_file){

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


void points2crd(vector<int> x, vector<int> y,vector<Point2f> &output_crd){

    for(int i=0;i<x.size();i++)
    {
        if (!x[i] && !y[i]) continue;

        Point p = Point2d((float)x[i], (float)y[i]);
        output_crd.push_back(p);
    }

}

void crd2points(vector<float> &x, vector<float> &y,vector<Point2f> &input_crd)
{
    x.clear();
    y.clear();

    for(int i=0;i<input_crd.size();i++)
    {
        x.push_back(input_crd[i].x);
        y.push_back(input_crd[i].y);
    }
}

pair<int,int> find_closest(vector<int> &x_of, vector<int> &y_of, int x, int y, float pixel_threshold)
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
    
    if (sqrt(min_dist) >= pixel_threshold)
        return make_pair(-1,-1);
    
    return make_pair(best_x,best_y);
}


void interpolate_next(vector<int> &x_prev, vector<int> &y_prev, 
                      vector<int> &x_of, vector<int> &y_of,
                      vector<float> &dist_threshold)
{

    for (int i = 0; i < x_prev.size(); i++)
    {
        if (x_prev[i] == -1)
            continue;

        pair<int,int> new_cord = find_closest(x_of,y_of,x_prev[i],y_prev[i],dist_threshold[i]);
        x_prev[i] = new_cord.first;
        y_prev[i] = new_cord.second;
    }
}

void get_statistics(vector<int> &x_itp, vector<int> &y_itp,
                    vector<float> &x_of, vector<float> &y_of,
                    vector<float> &maxim, vector<float> &mean,
                    vector<float> &minim, vector<float> &median,
                    vector<float> &percentiles)
{

    vector<float> distances;
    float dist = 0.0, total = 0.0;
    int n = x_itp.size();

    std::fill(percentiles.begin(),percentiles.end(),0);

    for (int i = 0; i < x_itp.size(); i++)
    {
        if (x_itp[i] == -1)
            continue;
        
        dist = (float) (((float)x_itp[i]-x_of[i])*((float)x_itp[i]-x_of[i]) + ((float)y_itp[i]-y_of[i])*((float)y_itp[i]-y_of[i]));        
        dist = sqrt(dist);
        
        for (int k = 0; k < percentiles_length; k++)
        {
             if (dist <= pixel_percentiles[k])
             {
                percentiles[k] = percentiles[k] + 1;
                break;
             } 
        }

        distances.push_back(dist);         
    }
    float found_sum = 0.0;

    for (int k = 0; k < percentiles.size(); k++)
       found_sum += percentiles[k];

    for (int k = 0; k < percentiles.size(); k++)
        percentiles[k] = (percentiles[k] / (found_sum + 0.00001)) * 100.0;
 
    if (!distances.size()) return;

    std::sort(distances.begin(),distances.end());
    int nk = distances.size(); 
    for (int k = 0; k < nk; k++) total += distances[k];

    maxim.push_back(distances.back());
    minim.push_back(distances[0]);
    mean.push_back(total/(float)nk);

    if (n % 2 == 0) median.push_back((distances[nk/2] + distances[nk/2 -1]) / 2.0);
    else median.push_back(distances[nk/2]);
     
}

int main(int argc, char ** argv) 
{

    int first_frame = -1, last_frame = -1;
    char *video_path = NULL, *out_file = NULL;
    bool verbose = false;
    int option = 0;
    vector<float> percentiles(5,0);

    while ((option = getopt(argc, argv,"s:e:f:o:v")) != -1) 
    {
        switch (option) {
             case 's' : first_frame = atoi(optarg);
                 break;
             case 'e' : last_frame = atoi(optarg);
                 break;
             case 'f' : video_path = optarg; 
                 break;
             case 'o' : out_file = optarg;
                 break;
             case 'v': verbose = true;
                 break;
             default:  
                 exit(EXIT_FAILURE);
        }
    }

    if (first_frame == -1) {cout<<"Invalid first frame argument"<<endl; return 0;}
    if (last_frame == -1) {cout<<"Invalid last frame argument"<<endl; return 0;}
    if (!video_path) {cout<<"Specify a video path!"<<endl; return 0;}
    if (!out_file) {cout<<"Specify results path!"<<endl; return 0;}

     cout<<"START "<<first_frame<<" END "<<last_frame<<" video path "<<video_path<<" OUT_FILE "<<out_file<<endl;

    Mat input, input_float;
    
    freopen(video_path,"r",stdin);
    Mat prev, current, fprev, fcurrent;
    vector<Point2f> tracking_points[2];
    vector<Point2f> custom_points[2];

    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(21,21);
    vector<int> x_truth;
    vector<int> y_truth;
    vector<float> minim, maxim, mean, median;
    vector<int> x_itp, y_itp;
    vector<float> x_of, y_of;
    vector<float> dist_threshold;

    float fx, fy, conf;

    string s_image1 = test_dir + "001" + format_img;
    string s_image2 = test_dir + "002" + format_img;

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
        string img = test_dir + number + format_img; 
        string output_image = output_dir + number + format_img;        
        input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);                                      

        if (i == 1)
        {
            frame_height = input.rows;
            frame_width = input.cols;
        }

        input.convertTo(input_float, CV_32F);
        //input_float *= (1/255.00);

        cin>>persons;
        
	x_truth.clear();
        y_truth.clear();

        for (int p = 0; p < persons; p++)
        {
            int valid_kp = 0;
            int lx = INT_MAX, rx = 0, ty = INT_MAX, by = 0;

	    /* Add all keypoints */
	    for (int k = 0; k < keypoints; k++)
	    {	      	    
                cin>>fx>>fy>>conf;
                if (conf < 0.01)
	            continue;
                
                valid_kp++;
                x_truth.push_back((int)fx);
                y_truth.push_back((int)fy);

                if (i == first_frame)
                {
                    lx = min(lx,(int)fx);
                    rx = max(rx,(int)fx);
                    ty = min(ty,(int)fy);
                    by = max(by,(int)fy);
                }
            }

            if (i == first_frame) 
            {
                for (int vkp = 0; vkp < valid_kp; vkp++)
                {
                    float max_p = (float) max(rx-lx,by-ty);
                    float max_f = (float) max(frame_height,frame_width);
                    //cout<<"TEST: "<<max_p<<" "<<max_f<<endl; 
                    dist_threshold.push_back(alpha * (max_p/max_f));
                    //cout<<dist_threshold.back()<<endl;
                }
            }
	}

        /* Set tracking points */
        if (i == first_frame)
        {
            fcurrent = input_float.clone();
            current = input;
            points2crd(x_truth, y_truth,tracking_points[1]);
            points2crd(x_truth, y_truth,custom_points[1]);
            custom_points[0].resize(custom_points[1].size());
            x_itp.assign(x_truth.begin(), x_truth.end());
            y_itp.assign(y_truth.begin(), y_truth.end());
            x_of.assign(x_truth.begin(), x_truth.end());
            y_of.assign(y_truth.begin(), y_truth.end());
        } 
        /* Draw OF keypoint estimates for frames (3,15) */
        if (i > first_frame && i < last_frame + 1)
        {
            vector<uchar> status;
            vector<float> err;
            vector<char> mystatus(custom_points[0].size());
            
            std::swap(tracking_points[1], tracking_points[0]);   
            std::swap(custom_points[1], custom_points[0]);
            cv::swap(fcurrent, fprev);
            cv::swap(current, prev);

            fcurrent = input_float.clone();
            current = input;
            interpolate_next(x_itp, y_itp,x_truth, y_truth,dist_threshold);

            calcOpticalFlowPyrLK(prev, current, tracking_points[0], tracking_points[1], 
                                 status, err, winSize, 5, termcrit, 0, 0.001);
            
            crd2points(x_of, y_of, tracking_points[1]);                        
            get_statistics(x_itp, y_itp, x_of, y_of, maxim, mean,minim,median,percentiles);
            
            if (verbose)
            {
                cout<<"MAXIMUM: "<<maxim.back()<<",  MINIMUM: "<<minim.back()<<", MEAN: "<<mean.back()<<", MEDIAN: "<<median.back()<<endl;   
                
                for (int k = 0; k < percentiles.size(); k++)
                    cout<<"Less than "<<pixel_percentiles[k]<<" pixels of error: "<<percentiles[k]<<"%"<<endl;
            }
            get_opt_flow(custom_points[0],custom_points[1],fprev, fcurrent, mystatus, 51);
            draw_all(input_float,x_truth,y_truth,tracking_points[1],custom_points[1], output_image);
        }     
        else    
            drawKeyPoints(input_float, x_truth, y_truth, output_image);
    }
    
    freopen(out_file,"w+",stdout);
    int sz = maxim.size();
    cout<<sz<<endl;
    /* Dump keypoints */
    for (int i = 0; i < sz; i++)
    {
        cout<<minim[i]<<endl;
        cout<<median[i]<<endl;
        cout<<maxim[i]<<endl;
        cout<<mean[i]<<endl;
    } 
    return 0;

}
