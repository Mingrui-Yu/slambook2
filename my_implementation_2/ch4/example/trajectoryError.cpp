#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include "sophus/se3.hpp"


typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;


std::string estimated_file_path = "estimated.txt";
std::string groundtruth_file_path = "groundtruth.txt";

TrajectoryType ReadTrajectory(std::string &file_path){
    std::fstream fin(file_path);
    TrajectoryType trajectory;
    double time, qx, qy, qz, qw, tx, ty, tz;

    if (!fin){
        std::cerr << "cannot open file " << file_path << "!" << std::endl;
    }

    while(!fin.eof()){
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        Eigen::Quaterniond q(qx, qy, qz, qw);
        Eigen::Vector3d t(tx, ty, tz);
        Sophus::SE3d T(q, t);

        trajectory.push_back(T);
    }

    return trajectory;
}






int main(int argc, char **argv){

    TrajectoryType estimated_trajectory = ReadTrajectory(estimated_file_path);
    TrajectoryType groundtruth_trajectory = ReadTrajectory(groundtruth_file_path);

    if (estimated_trajectory.size() != groundtruth_trajectory.size()){
        std::cerr << "the length of estimate_trajectory != length of groundtruth trajectory!" << std::endl;
    }

    double total_error = 0, error;

    for(size_t i = 0; i < estimated_trajectory.size(); i++){
        Sophus::SE3d T_estimated = estimated_trajectory[i];
        Sophus::SE3d T_groundtruth = groundtruth_trajectory[i];

        error = (T_groundtruth.inverse() * T_estimated).log().norm();
        total_error += error * error;
    }

    double rmse = sqrt(total_error / estimated_trajectory.size());
    
    std::cout << "RMSE = " << rmse << std::endl;

    return 0;
}



