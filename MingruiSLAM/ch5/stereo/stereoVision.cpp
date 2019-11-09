#include "showPointCloud.h"

using namespace std;
using namespace Eigen;

string left_file = "left.png";
string right_file = "right.png";

int main(int argc, char **argv)
{
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线
    double b = 0.573;


    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);


    // 调用OpenCV实现的SGBM(Semi-Global Batch Matching)算法计算左右视差
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0/16.0f);


    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
    // 遍历图像中所有点，计算每个点在世界坐标系中的坐标
    for (int v=0; v<left.rows; v++)
    {
        for (int u=0; u<left.cols; u++)
        {
            if (disparity.at<float>(v,u) <= 0 || disparity.at<float>(v,u) >= 96.0) continue;
            
            Vector4d point(0, 0, 0, left.at<uchar>(v,u)/255.0); // 前三维为xyz，第四维为color

            // 根据视差，计算point在世界坐标系中的位置
            double Z = fx * b / disparity.at<float>(v,u);
            point[0] = (u-cx)/fx * Z;
            point[1] = (v-cy)/fy * Z;
            point[2] = Z;

            pointcloud.push_back(point);
        }
    }

    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    // 画出点云
    showPointCloud(pointcloud);

    return 0;
}


