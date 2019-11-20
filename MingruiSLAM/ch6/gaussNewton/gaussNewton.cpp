#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;  // Opencv随机数产生器


    // 生成数据
    vector<double> x_data, y_data;
    for (int i=0; i<N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        double y = exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma);
        y_data.push_back(y);
    }


    // 开始Gauss-Newton迭代
    int iter;
    double cost=0, lastCost = 0;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    while(1)
    {
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero(); 
        cost = 0;

        for (int i=0; i<N; i++)
        {
            double x = x_data[i];
            double y = y_data[i];

            double error = y - exp(ae*x*x + be*x + ce);

            double de_da = - x*x*exp(ae*x*x + be*x +ce);
            double de_db = - x*exp(ae*x*x + be*x +ce);
            double de_dc = - exp(ae*x*x + be*x +ce);

            Vector3d J;
            J[0] = de_da;
            J[1] = de_db;
            J[2] = de_dc;

            H = H + J * J.transpose();
            b = b - J * error;

            cost = cost + error*error;
        }   

        // 求解线性方程 Hx=b
        Vector3d dx= H.ldlt().solve(b);
        if(isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost)
        {
            cout << "cost >= lastCost, break!" << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        cout << "iter = " << iter << ", total cost = " << cost << ", \t\t update = " << dx.transpose() 
             << ", \t\t estimated paras = " << ae << ", " << be << ", " << ce << endl; 

        
        if (fabs(cost - lastCost) < 1e-8)
        {
            break;
        }

        lastCost = cost;
        iter += 1;       
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2-t1);

    cout << endl << "Estimated paras a, b, c = " << ae << ", " << be << ", " << ce << endl; 
    cout << endl << "Solve time cost = " << time_used.count() << "s" << endl;



    return 0;
}