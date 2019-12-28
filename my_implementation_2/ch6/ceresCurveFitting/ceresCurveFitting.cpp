#include <iostream>
#include <chrono>

#include <Eigen/Core>  // Eigen 核心部分
#include <Eigen/Dense>  // 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Geometry>  // Eigen/Geometry 模块提供了各种旋转和平移的表示

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <ceres/ceres.h>



struct CURVE_FITTING_COST{

    CURVE_FITTING_COST(double t, double y): _t(t), _y(y) {}

    template<typename T>
    bool operator() (const T *const x, T *residual) const {
        residual[0] = T(_y) - ceres::exp(x[0]*T(_t)*T(_t) + x[1]*T(_t) + x[2]);
        return true;
    }

    const double _t, _y;
};

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


    double x[3] = {a_e, b_e, c_e};
    
    // 构建最小二乘问题
    ceres::Problem problem;
    for(int i = 0; i < num_data; i++){
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new 
                CURVE_FITTING_COST(t_data[i], y_data[i])),
            nullptr,
            x
        );
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast <std::chrono::duration<double>> (t2 - t1);
    std::cout << "time_cost = " << time_used.count() << "second" << std::endl; 

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "estimated_parameters = " << x[0] << " " << x[1] << " " << x[2] << std::endl;
    return 0;
}
