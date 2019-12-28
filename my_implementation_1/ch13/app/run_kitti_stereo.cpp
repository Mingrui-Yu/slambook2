#include <gflags/gflags.h>
#include <myslam/visual_odometry.h>

DEFINE_string(config_file, "./config/default.yaml", "config file path");

int main(int argc, char **argv){
    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);
    vo->Run();

    std::string save_file = "./map/all_key_poses.txt";
    vo->SaveAllKeyPoseInTXT(save_file);

    // vo->ReadAndDrawTrajectory(save_file);

    return 0;
}