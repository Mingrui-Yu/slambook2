#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char **argv)
{
    string image_file = "distorted.png";
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file, 0);
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1) ;

    for(int v=0; v<rows; v++)
    {
        for(int u=0; u<cols; u++)
        {
            double x = (u - cx)/fx, y = (v-cy)/fy;
            double r = sqrt(x*x + y*y);

            double x_distorted = x*(1 + k1*r*r + k2*pow(r,4)) + 2*p1*x*y + p2*(r*r + 2*x*x);
            double y_distorted = y*(1 + k1*r*r + k2*pow(r,4)) + p1*x*y + 2*p2*(r*r + 2*x*x);

            int u_distorted = fx * x_distorted + cx + 0.5;
            int v_distorted = fy * y_distorted + cy + 0.5;

            if ((u_distorted>=0 && u_distorted<cols) && (v_distorted>=0 && v_distorted<rows))
            {
                image_undistort.at<uchar>(v,u) = image.at<uchar>(v_distorted, u_distorted);
            }
            else
            {
                image_undistort.at<uchar>(v,u) = 0;
            }
            
        }
    }

    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey(0);

    return 0;
}