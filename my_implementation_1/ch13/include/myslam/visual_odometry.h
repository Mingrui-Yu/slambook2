#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/backend.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"
#include "myslam/dataset.h"
#include "myslam/map.h"


namespace myslam{

/**
 * VO 对外借口
 */
class VisualOdometry{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualOdometry> Ptr;

    // constructor with config file
    VisualOdometry(std::string &config_path);

    /**
     * do initialization things before run
     */
    bool Init();

    /**
     * start VO in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    // 获取前端状态
    FrontendStatus GetFrontendStatus() const{
        return frontend_->GetStatus();
    }

    // 将map中的所有frame对应的pose保存到txt文件中
    void SaveAllKeyPoseInTXT(std::string &save_file);

    // 读取保存所有keyPose的txt文件，并且展示trajectory
    void ReadAndDrawTrajectory(std::string &trajectory_file); 

private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    //dataset
    Dataset::Ptr dataset_ = nullptr;

    //std::unordered_map<unsigned long, SE3> all_key_poses;
};

}  // namespace myslam


#endif