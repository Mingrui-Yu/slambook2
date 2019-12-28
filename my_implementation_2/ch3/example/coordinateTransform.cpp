#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;


int main(int argc, char **argv){

    Eigen::Vector3d p1(0.5, 0, 0.2);

    Eigen::Quaterniond q1(0.35, 0.2, 0.3, 0.1);
    q1.normalize();
    Eigen::Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
    q2.normalize();

    Eigen::Vector3d t1(0.3, 0.1, 0.1);
    Eigen::Vector3d t2(-0.1, 0.5, 0.3);

    Eigen::Isometry3d T_r1w(q1);
    T_r1w.pretranslate(t1);
    Eigen::Isometry3d T_wr1 = T_r1w.inverse();

    Eigen::Isometry3d T_r2w(q2);
    T_r2w.pretranslate(t2);
    Eigen::Isometry3d T_wr2 = T_r2w.inverse();

    Eigen::Vector3d p2 = T_wr2.inverse() * T_wr1 * p1;

    cout << "p2 = " << p2.transpose() << endl;

    return 0;
}