#include <iostream>

#include <Eigen/Core>  // Eigen 核心部分
#include <Eigen/Dense>  // 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Geometry>  // Eigen/Geometry 模块提供了各种旋转和平移的表示

#include "sophus/se3.hpp"

#include <opencv2/core.hpp>  // 核心功能，包括基本数据结构
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>  // 高层GUI图像交互
#include <opencv2/features2d.hpp>  // 特征提取、描述、匹配

// 相机内参
const double f = 521, cx = 325.1, cy = 249.7;

bool ORBFeatureDetectAndMatch(
    cv::Mat &img_1, cv::Mat &img_2,
    std::vector<cv::KeyPoint> &kps_1, 
    std::vector<cv::KeyPoint> &kps_2,
    std::vector<cv::DMatch> &good_matches
);



bool PoseEstimation2d2d(
    std::vector<cv::KeyPoint> &kps_1, 
    std::vector<cv::KeyPoint> &kps_2,
    std::vector<cv::DMatch> &good_matches,
    cv::Mat &R,
    cv::Mat &t
){
    std::vector<cv::Point2f> points_1;
    std::vector<cv::Point2f> points_2;
    for(size_t i = 0; i < good_matches.size(); i++){
        points_1.push_back(kps_1[good_matches[i].queryIdx].pt);
        points_2.push_back(kps_2[good_matches[i].trainIdx].pt);
    }

    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points_1, points_2, CV_FM_8POINT);
    std::cout << "fundamental matrix = \n" << fundamental_matrix << std::endl;

    const cv::Point2d principal_point(cx, cy);
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points_1, points_2, f, principal_point);
    std::cout << "essential matrix = \n" << essential_matrix << std::endl;

    cv::recoverPose(essential_matrix, points_1, points_2, R, t, f, principal_point);
}






int main(int argc, char **argv){
    std::string file_path_1 = argv[1];
    std::string file_path_2 = argv[2];

    cv::Mat img_1 = cv::imread(file_path_1, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(file_path_2, cv::IMREAD_COLOR);

    if (img_1.empty() || img_2.empty()){
        std::cerr << "cannot find image: " << file_path_1 << " or " << file_path_2 << std::endl;
        return -1;
    }

    // ORB feature detection and match
    std::vector<cv::KeyPoint> kps_1, kps_2;
    std::vector<cv::DMatch> good_matches;
    ORBFeatureDetectAndMatch(img_1, img_2, kps_1, kps_2, good_matches);

    // show the keypoints and matches
    cv::Mat img_matches;
    cv::drawMatches(img_1, kps_1, img_2, kps_2, good_matches, img_matches);
    cv::imshow("matches between img_1 & img_2", img_matches);
    
    cv::Mat R, t;
    PoseEstimation2d2d(kps_1, kps_2, good_matches, R, t);
    std::cout << "R = \n" << R << std::endl;
    std::cout << "t = \n" << t << std::endl;


    cv::waitKey(0);
    return 0;
}






bool ORBFeatureDetectAndMatch(
    cv::Mat &img_1, cv::Mat &img_2,
    std::vector<cv::KeyPoint> &kps_1, 
    std::vector<cv::KeyPoint> &kps_2,
    std::vector<cv::DMatch> &good_matches
    ){
    // orb提取特征点和描述子
    cv::Mat descriptor_1, descriptor_2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detectAndCompute(img_1, cv::noArray(), kps_1, descriptor_1);
    orb->detectAndCompute(img_2, cv::noArray(), kps_2, descriptor_2);

    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    // std::vector<cv::DMatch> match;
    // matcher->match(descriptor_1, descriptor_2, match);


    // // --第四步：匹配点对 筛选
    // auto min_max = minmax_element(match.begin(), match.end(),
    //             [] (const cv::DMatch &m1, const cv::DMatch &m2) {return m1.distance < m2.distance;});
    // double min_dist = min_max.first->distance;
    // double max_dist = min_max.second->distance;

    // for (int i=0; i<descriptor_1.rows; i++){
    //     if(match[i].distance <= std::max(2*min_dist, 30.0)){
    //         good_matches.push_back(match[i]);
    //     }
    // }


    // 进行匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = 
        cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(descriptor_1, descriptor_2, matches);

    double min_dist = 10000;
    for (auto &m: matches){
        if (m.distance < min_dist){
            min_dist = m.distance;
        }
    }

    std::cout << "min_dist = " << min_dist << std::endl;
    
    for(size_t i = 0; i < matches.size(); i++){
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }
    std::cout << "have found " << good_matches.size() << " good matches. " << std::endl;
    return true;
}