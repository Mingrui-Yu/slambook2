#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<opencv2/core/eigen.hpp>
#include <chrono>
#include <sophus/se3.hpp>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;
using namespace cv;
using namespace Eigen;

// 相机内参
Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void pose_estimation_3d3d(const vector<Point3d> &pts1, const vector<Point3d> &pts2,
                          Mat &R, Mat &t);

void bundelAdjustment(const vector<Point3d> &pts1, const vector<Point3d> &pts2, 
                      Mat &R, Mat &t);


// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p);














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
  // for (DMatch m:matches) {
  //   cout << keypoints_1[m.queryIdx].pt.x << " " << keypoints_1[m.queryIdx].pt.y << endl;
  // }



  // 读取深度图，建立3D点
  Mat img_depth_1 = imread("../1_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  Mat img_depth_2 = imread("../2_depth.png", CV_LOAD_IMAGE_UNCHANGED);

  vector<Point3d> points_1;
  vector<Point3d> points_2;

  for (DMatch m:matches){
    ushort d1 = img_depth_1.at<unsigned short>((int)keypoints_1[m.queryIdx].pt.y, (int)keypoints_1[m.queryIdx].pt.x);
    ushort d2 = img_depth_2.at<unsigned short>((int)keypoints_2[m.trainIdx].pt.y, (int)keypoints_2[m.trainIdx].pt.x);

    if (d1==0 || d2==0) continue;

    float dd2 = d2 / 5000.0;
    float dd1 = d1 / 5000.0;
    
    Point2d p1_2d = pixel2cam(keypoints_1[m.queryIdx].pt);
    Point2d p2_2d = pixel2cam(keypoints_2[m.trainIdx].pt);

    points_1.push_back(Point3d(p1_2d.x * dd1, p1_2d.y*dd1, dd1));
    points_2.push_back(Point3d(p2_2d.x * dd2, p2_2d.y*dd2, dd2));
  }
  cout << "valid 3d-3d pairs: " << points_1.size() << endl;

  Mat R, t;
  cout << "开始SVD求解 ......" << endl;
  pose_estimation_3d3d(points_1, points_2, R, t);
  cout << "ICP via SVD results: " << endl;
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;
  cout << endl;

  cout << "开始BA求解 ......" << endl;
  bundelAdjustment(points_1, points_2, R, t);
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;
  cout << endl;

  return 0;
} 











class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};


class EdgeProjectXYZRGBDPoseOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point): _point(point){}

  virtual void computeError() override {
    const VertexPose *pose = static_cast<const VertexPose *>(_vertices[0]);
    _error = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override {
    const VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d xyz_trans = T * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = - Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }

  bool read(istream &in) {}
  bool write(ostream &out) const {}

  protected:
  Eigen::Vector3d _point;
};


void bundelAdjustment(const vector<Point3d> &pts2, const vector<Point3d> &pts1, 
                      Mat &R, Mat &t){
  typedef g2o::BlockSolverX BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

  auto solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  VertexPose *vertex_pose = new VertexPose();
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  for(size_t i=0; i < pts1.size(); i++){
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2[i].x,
    pts2[i].y, pts2[i].z));
    edge->setId(i);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }

  optimizer.initializeOptimization();
  optimizer.optimize(10);
  cout << "T=\n" << vertex_pose->estimate().matrix() << endl;

  Eigen::Matrix3d R_ = vertex_pose->estimate().rotationMatrix();
  Eigen::Vector3d t_ = vertex_pose->estimate().translation();

  cv::eigen2cv(R_, R);
  cv::eigen2cv(t_, t);
}















// 注意pts2和pts1顺序
void pose_estimation_3d3d(const vector<Point3d> &pts2, const vector<Point3d> &pts1,
                          Mat &R, Mat &t){
  
  assert(pts1.size() == pts2.size());
  int N = pts1.size();

  Point3d p1, p2;
  for (int i=0; i<N; i++){
    p1 += pts1[i];
    p2 += pts2[i];
  }

  p1 = p1 / N;
  p2 = p2 / N;

  vector<Point3d> q1(N), q2(N);
  for (int i=0; i<N; i++){
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  Matrix3d W = Matrix3d::Zero();
  for (int i=0; i<N; i++){
    Vector3d q1_eig(q1[i].x, q1[i].y, q1[i].z);
    Vector3d q2_eig(q2[i].x, q2[i].y, q2[i].z);
    W += q1_eig * q2_eig.transpose();
  }
  cout << "W = " << W << endl;

  Eigen::JacobiSVD<Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  cout << "U=" << U << endl;
  cout << "V=" << V << endl;

  Eigen::Matrix3d R_eig = U * V.transpose();
  if (R_eig.determinant() < 0){
    R_eig = -R_eig;
  }

  Vector3d p1_eig(p1.x, p1.y, p1.z);
  Vector3d p2_eig(p2.x, p2.y, p2.z);
  Eigen::Vector3d t_eig = p1_eig - R_eig*p2_eig;

  cv::eigen2cv(R_eig, R);
  cv::eigen2cv(t_eig, t);
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






Point2d pixel2cam(const Point2d &p) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

