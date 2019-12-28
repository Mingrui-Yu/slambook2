#include <iostream>
#include <chrono>

#include <Eigen/Core>  // Eigen 核心部分
#include <Eigen/Dense>  // 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Geometry>  // Eigen/Geometry 模块提供了各种旋转和平移的表示

#include <opencv2/opencv.hpp>

int main(int argc, char **argv){

    double a_e = 2.0, b_e = -1.0, c_e = 5.0;  // 待拟合参数 及估计初值
    double a = 1.0, b = 2.0, c = 1.0;  // 待拟合参数 实际值
    cv::RNG rng;  // OpenCV 随机数产生器
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;

    // 生成数据
    int num_data = 100;
    std::vector<double> t_data, y_data;
    
    for(int i = 0; i < num_data; i++){
        double t = i / 100.0;
        double y = exp(a*t*t + b*t + c) + rng.gaussian(w_sigma * w_sigma);
        t_data.push_back(t);
        y_data.push_back(y);
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // 迭代-优化
    Eigen::Vector3d x(a_e, b_e, c_e);
    int num_iters = 20;
    double cost = 0, lastCost = 0;
    
    for(int iter = 0; iter < num_iters; iter++){
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        cost = 0;
        for (int i = 0; i < num_data; i++){
            double t = t_data[i];
            double y = y_data[i];
            double y_e = exp(x[0]*t*t + x[1]*t + x[2]);

            double error = y_e - y;
            Eigen::Matrix<double, 1, 3> J;
            J[0] = t * t * exp(x[0]*t*t + x[1]*t + x[2]); 
            J[1] = t * exp(x[0]*t*t + x[1]*t + x[2]);
            J[2] = exp(x[0]*t*t + x[1]*t + x[2]);

            H += inv_sigma * inv_sigma * J.transpose() * J;
            g += - inv_sigma * inv_sigma * J * error;
            cost += error * error;
        }

        if (iter >= 1 && cost >= lastCost){
            std::cout << "cost >= lastCost, break." << std::endl; 
            break;
        }
        lastCost = cost;

        Eigen::Vector3d dx = H.ldlt().solve(g);
        if (dx.norm() < 1e-6){
            std::cout << "||dx|| < 1e-6, break." << std::endl;
            break;
        }
        x += dx;

        std::cout << "iter = " << iter << " , estimated parameters = " << x.transpose() 
                  << " , cost = " << cost << std::endl;
    }
    
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
    
    std::cout << "final , estimated parameters = " << x.transpose() 
                  << " , cost = " << cost << std::endl;
    std::cout << "time_cost = " << time_used.count() << "second" << std::endl; 


    return 0;
}
