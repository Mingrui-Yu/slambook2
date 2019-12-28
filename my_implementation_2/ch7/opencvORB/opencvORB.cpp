#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
// #include "opencv2/xfeatures2d.hpp"

int main(int argc, char **argv){

    std::string file_path_1 = argv[1];
    std::string file_path_2 = argv[2];

    cv::Mat img_1 = cv::imread(file_path_1, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(file_path_2, cv::IMREAD_COLOR);

    if (img_1.data == nullptr || img_2.data == nullptr){
        std::cerr << "cannot find image: " << file_path_1 << " or " << file_path_2 << std::endl;
        return -1;
    }

    // cv::imshow("img_1", img_1);
    // cv::imshow("img_2", img_2);
    // cv::waitKey(0);

    // 检测Oriented Fast角点位置
    std::vector<cv::KeyPoint> kps_1, kps_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    detector->detect(img_1, kps_1);
    detector->detect(img_2, kps_2);

    // 根据角点位置计算BRIEF描述子
    cv::Mat descriptor_1, descriptor_2;
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    descriptor->compute(img_1, kps_1, descriptor_1);
    descriptor->compute(img_2, kps_2, descriptor_2);

    // 画出keypoints
    cv::Mat img_kps_1, img_kps_2;
    cv::drawKeypoints(img_1, kps_1, img_kps_1);
    cv::drawKeypoints(img_2, kps_2, img_kps_2);
    cv::imshow("img_1 with ORB features", img_kps_1);
    cv::imshow("img_2 with ORB features", img_kps_2);
    cv::waitKey(0);

    // 进行匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = 
        cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(descriptor_1, descriptor_2, matches);

    double min_dist = 15;
    for (auto &m: matches){
        if (m.distance < min_dist){
            min_dist = m.distance;
        }
    }
    std::vector<cv::DMatch> good_matches;
    for(size_t i = 0; i < matches.size(); i++){
        if (matches[i].distance < std::max(2 * min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }

    std::cout << "number of good matches = " << good_matches.size() << std::endl;

    cv::Mat img_matches;
    cv::drawMatches(img_1, kps_1, img_2, kps_2, good_matches, img_matches);
    cv::imshow("matches between img_1 & img_2", img_matches);
    cv::waitKey(0);




    return 0;
}

