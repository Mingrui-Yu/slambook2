#include <iostream>
#include <ctime>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 50

int main( int argc, char** argv)
{
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero(); //初始化为零

    matrix_33 = Eigen::Matrix3d::Random();
    cout << matrix_33 << endl << endl;

    cout << matrix_33.transpose() << endl;
    cout << matrix_33.sum() << endl;
    cout << matrix_33.trace() << endl;
    cout << 10*matrix_33 << endl;
    cout << 10*matrix_33.inverse() << endl;
    cout << matrix_33.determinant() << endl << endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
    cout << "Eigen Value = " << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = " << eigen_solver.eigenvectors() << endl;

}