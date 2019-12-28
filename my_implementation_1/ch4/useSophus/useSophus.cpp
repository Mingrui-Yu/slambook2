#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;


int main(int argc, char **argv)
{
    // 定义旋转矩阵（由旋转向量-旋转矩阵）
    Matrix3d R = AngleAxisd(M_PI/2, Vector3d(0, 0, 1)).toRotationMatrix();
    // 转换成四元数
    Quaterniond q(R);

    // SO3 可以从旋转矩阵构造，也可以从四元数构造
    Sophus::SO3d SO3_R(R);
    Sophus::SO3d SO3_q(q);

    cout .precision(3);
    cout << "旋转矩阵R：\n" << R << endl;
    cout << "由旋转矩阵构造：\n" << SO3_R.matrix() << endl;
    cout << "由四元数构造：\n" << SO3_q.matrix() << endl;
    cout << endl;


    // 李群——李代数
    Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;

    // hat 为向量——反对称矩阵
    Matrix3d so3_hat = Sophus::SO3d::hat(so3);
    cout << "so3 hat = \n" << so3_hat << endl;

    // vee 为反对称矩阵——向量
    Vector3d so3_hat_vee = Sophus::SO3d::vee(so3_hat);
    cout << "so3 hat vee =" << so3_hat_vee.transpose() << endl;
    cout << endl;


    // 增量扰动模型的更新
    Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;
    cout << "**********************************************" << endl << endl;


    // 对SE(3)操作
    Vector3d t(1, 0, 0);
    Sophus::SE3d SE3_Rt(R, t);
    Sophus::SE3d SE3_qt(q, t);
    cout << "由R+t构造：\n" << SE3_Rt.matrix() << endl;
    cout << "由q+t构造：\n" << SE3_qt.matrix() << endl;

    cout << "SE3的平移部分： \n" << SE3_Rt.translation() << endl;

    // 李代数se(3)是一个六维向量
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl;

    // hat 为向量——反对称矩阵
    Matrix4d se3_hat = Sophus::SE3d::hat(se3);
    cout << "se3 hat = \n" << se3_hat << endl;

    // vee 为反对称矩阵——向量
    Vector6d se3_hat_vee = Sophus::SE3d::vee(se3_hat);
    cout << "se3 hat vee =" << se3_hat_vee.transpose() << endl;
    cout << endl;


    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0, 0) = 1e-4d;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3 updated = \n" << SE3_updated.matrix() << endl;











    return 0;
}