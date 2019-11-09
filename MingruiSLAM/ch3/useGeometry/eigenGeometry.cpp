#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#define MATRIX_SIZE 10

int main(int argc, char** argv)
{
    Eigen::Quaterniond q1 = Eigen::Quaterniond(0.55, 0.3, 0.2, 0.2);
    q1.normalize();
    Eigen::Vector3d t1 = Eigen::Vector3d (0.7, 1.1, 0.2);

    Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();
    T1.rotate(q1);
    T1.pretranslate(t1);

    Eigen::Quaterniond q2 = Eigen::Quaterniond(-0.1, 0.3, -0.7, 0.2);
    q2.normalize();
    Eigen::Vector3d t2 = Eigen::Vector3d (-0.1, 0.4, 0.8);

    Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();
    T2.rotate(q2);
    T2.pretranslate(t2);

    Eigen::Vector3d p1 (0.5,-0.1,0.2);
    Eigen::Vector3d p2;

    p2 = T2 * (T1.inverse()) * p1;
    
    
    cout .precision(3);
    cout << "p2 = " << p2.transpose() << endl;



    




    return 0;
}

