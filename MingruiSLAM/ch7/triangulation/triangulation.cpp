#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 相机内参
const Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void pose_estimation_2d2d(
  std::vector<KeyPoint> keypoints_1,
  std::vector<KeyPoint> keypoints_2,
  std::vector<DMatch> matches,
  Mat &R, Mat &t);

void triangulation(const vector<KeyPoint> &keypoints_1, const vector<KeyPoint> &keypoints_2, 
                   const vector<DMatch> &matches, const Mat &R, const Mat &t, 
                   vector<Point3d> &points);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p);

/// 作图用
inline cv::Scalar get_color(float depth) {
  float up_th = 20, low_th = 5, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * (depth-low_th) / th_range, 0, 255 * (1 - (depth-low_th) / th_range));
}




int main(int argc, char **argv){

  // 读取图像
  Mat img_1 = imread("../1.png", CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread("../2.png", CV_LOAD_IMAGE_COLOR);
  assert(img_1.data && img_2.data);
  cout << "读取图像 完成！" << endl;


  // 特征点匹配
  cout << "开始特征点匹配 ......" << endl;
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "特征点匹配 完成！ 一共找到了" << matches.size() << "组匹配点" << endl << endl;


  // 估计两张图像间的运动
  cout << "开始估计两张图像间的运动：计算R t" << endl;
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);
  cout << "估计R t 完成！" << endl << endl;

  // 三角化
  cout << "开始三角化 ......" << endl;
  vector<Point3d> points;
  triangulation(keypoints_1, keypoints_2, matches, R, t, points);
  cout << "三角化 完成！" << endl << endl;

  // 验证三角化与特征点的重投影关系
  Mat img1_plot = img_1.clone();
  Mat img2_plot = img_2.clone();
  for (int i = 0; i < matches.size(); i++) {
    // 第一个图
    float depth1 = points[i].z;
    cout << "depth: " << depth1 << endl;
    Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt);
    cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

    // 第二个图
    Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
    float depth2 = pt2_trans.at<double>(2, 0);
    cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
  }
  cv::imshow("img 1", img1_plot);
  cv::imshow("img 2", img2_plot);
  cv::waitKey();
  
  return 0;
}









void triangulation(const vector<KeyPoint> &keypoints_1, const vector<KeyPoint> &keypoints_2, 
                   const vector<DMatch> &matches, const Mat &R, const Mat &t, 
                   vector<Point3d> &points){
  
  vector<Point2f> points1, points2;

  for (int i=0; i < (int) matches.size(); i++){
    points1.push_back(pixel2cam(keypoints_1[matches[i].queryIdx].pt));
    points2.push_back(pixel2cam(keypoints_2[matches[i].trainIdx].pt));
  }

  Mat T1 = (Mat_<float>(3,4) << 1,0,0,0,
                               0,1,0,0,
                               0,0,1,0);

  
  Mat T2;
  hconcat(R, t, T2);
  cout << "T2:" << T2 << endl;


  Mat points_4d;
  cv::triangulatePoints(T1, T2, points1, points2, points_4d);

  // 讲points_4d转换成非齐次坐标
  for (int i = 0; i < points_4d.cols; i++){
    Mat x = points_4d.col(i);
    x = x / x.at<float>(3,0);
    Point3d p(x.at<float>(0,0), x.at<float>(1,0), x.at<float>(2,0));
    points.push_back(p);
  }
}













void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches){

    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // --第一步：检测Oriented Fast角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    cout << "--第一步完成：检测Oriented Fast角点位置" << endl;

    // --第二步：根据角点位置计算BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    cout << "--第二步完成：根据角点位置计算BRIEF描述子" << endl;

    // -- 第三步：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    cout << "--第三步完成：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离" << endl;

    // --第四步：匹配点对 筛选
    auto min_max = minmax_element(match.begin(), match.end(),
                [] (const DMatch &m1, const DMatch &m2) {return m1.distance < m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    for (int i=0; i<descriptors_1.rows; i++){
        if(match[i].distance <= max(2*min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }
    cout << "--第四步完成：匹配点对 筛选" << endl;
}


void pose_estimation_2d2d(
  std::vector<KeyPoint> keypoints_1,
  std::vector<KeyPoint> keypoints_2,
  std::vector<DMatch> matches,
  Mat &R, Mat &t){

    // 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i=0; i < (int) matches.size(); i++){
      points1.push_back(keypoints_1[matches[i].queryIdx].pt);
      points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "--fundamental_matrix is " << endl << fundamental_matrix << endl;

    // 计算本质矩阵
    Point2d principal_point;
    principal_point.x = K.at<double>(0,2);
    principal_point.y = K.at<double>(1,2);
    double focal_length = K.at<double>(1,2);
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout <<  "--essential_matrix is " << endl << essential_matrix << endl;

    //计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout <<  "--homography_matrix is " << endl << homography_matrix << endl;

    // 从本质矩阵中恢复R t
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "--R = " << endl << R << endl;
    cout << "--t = " << t.t() << endl;
  }

  Point2d pixel2cam(const Point2d &p) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}