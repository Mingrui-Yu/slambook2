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

void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d,
                                 const Mat &K, Sophus::SE3d &pose);
void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d,
                         const Mat &K, Sophus::SE3d &pose);


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

  vector<Point3d> points_3d;
  vector<Point2d> points_2d;

  for (DMatch m:matches){
    // ushort d = img_depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d = img_depth_1.at<unsigned short>((int)keypoints_1[m.queryIdx].pt.y, (int)keypoints_1[m.queryIdx].pt.x);
    if (d==0) continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt);
    points_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    points_2d.push_back(keypoints_2[m.trainIdx].pt);
  }
  cout << "Total number of valid 3d-2d pairs: " << points_3d.size() << endl;


  cout << "开始PnP求解 ......" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat R, r, t;
  solvePnP(points_3d, points_2d, K, Mat(), r, t, false);
  cv::Rodrigues(r, R);
  cout << "R = " << endl << R << endl;
  cout << "t = " << t.t() << endl;
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
  cout << endl;


  cout << "开始手写G-N优化 ......" << endl;
  VecVector3d points_3d_eig;
  VecVector2d points_2d_eig;
  for(size_t i=0; i<points_3d.size(); i++){
    Point3d p_3d = points_3d[i];
    Point2d p_2d = points_2d[i];
    points_3d_eig.push_back(Vector3d(p_3d.x, p_3d.y, p_3d.z));
    points_2d_eig.push_back(Vector2d(p_2d.x, p_2d.y));
  }

  Matrix3d R_eig;
  Vector3d t_eig;
  cv2eigen(R, R_eig);
  cv2eigen(t, t_eig);
  Sophus::SE3d pose_gn(R_eig, t_eig);
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(points_3d_eig, points_2d_eig, K, pose_gn);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "pose by g-n: " << endl << pose_gn.matrix() << endl;
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl << endl;


  cout << "开始G2O优化 ......" << endl;
  Sophus::SE3d pose_g2o(R_eig, t_eig);
  bundleAdjustmentG2O(points_3d_eig, points_2d_eig, K, pose_g2o);
  cout << "pose by g2o: " << endl << pose_gn.matrix() << endl;


  return 0;
} 




void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d,
                                 const Mat &K, Sophus::SE3d &pose){

  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 10;
  double cost = 0, lastCost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter=0; iter<iterations; iter++){
    Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();
    cost = 0;

    for (int i=0; i<points_3d.size(); i++){
      Vector3d pc = pose * points_3d[i];
      double inv_z = 1.0 / pc[2];
      double inv_z2 = inv_z * inv_z;
      Vector2d proj (fx*pc[0]*inv_z + cx,
                     fy*pc[1]*inv_z + cy);
      Vector2d e = points_2d[i] - proj;

      cost += e.squaredNorm();

      Matrix<double, 2, 6> J;
      J << -fx * inv_z,
        0,
        fx * pc[0] * inv_z2,
        fx * pc[0] * pc[1] * inv_z2,
        -fx - fx * pc[0] * pc[0] * inv_z2,
        fx * pc[1] * inv_z,
        0,
        -fy * inv_z,
        fy * pc[1] * inv_z2,
        fy + fy * pc[1] * pc[1] * inv_z2,
        -fy * pc[0] * pc[1] * inv_z2,
        -fy * pc[0] * inv_z;

      H += J.transpose() * J;
      b +=  - J.transpose() * e;
    }

    Vector6d dx;
    dx = H.ldlt().solve(b);

    if (isnan(dx[0])){
      cout << "result is nan!" << endl;
    }

    if (iter > 0 && cost > lastCost){
      cout << "the cost is larger than last cost!";
      break;
    }

    pose *= Sophus::SE3d::exp(dx);
    lastCost = cost;

    cout << "iteration " << iter << " cost=" << cost << endl;

    if (dx.norm() < 1e-6){
      cout << "have converge." << endl;
      break;
    }
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






Point2d pixel2cam(const Point2d &p) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}












class VertexPose: public g2o::BaseVertex<6, Sophus::SE3d>{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};


class EdgeProjection: public g2o::BaseUnaryEdge<2, Vector2d, VertexPose>{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection(const Vector3d &pos, const Matrix3d &K): _pos3d(pos), _K(K){}

  virtual void computeError() override{
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel = pos_pixel / pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}

  private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};


void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d,
                         const Mat &K, Sophus::SE3d &pose){
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  VertexPose *vertex_pose = new VertexPose();
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  Matrix3d K_eigen;
  cv2eigen(K, K_eigen);

  for(size_t i=0; i < points_3d.size(); i++){
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(i);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(p2d);
    edge->setInformation(Matrix2d::Identity());
    optimizer.addEdge(edge);
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  pose = vertex_pose->estimate();
}