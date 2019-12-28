#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"

namespace myslam{

// 前向声明
struct MapPoint;
struct Feature;

/**
 * 帧
 * 每一帧分配独立ID，关键帧分配关键帧ID
 */
struct Frame{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;  // id of this frame
    unsigned long keyframe_id_ = 0;  // id of keyframe
    bool is_keyframe_ = false;  // if is the keyframe
    double time_stamp_;  // 时间戳，暂不使用
    SE3 pose_;
    std::mutex pose_mutex_;  // pose 数据锁
    cv::Mat left_img_, right_img_;  // stereo images

    // extract features in left image
    std::vector<std::shared_ptr<Feature>> features_left_;
    // extract features in right image
    std::vector<std::shared_ptr<Feature>> features_right_;

public:
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, 
            const Mat &left, const Mat &right);

    // set and get pose, thread safe
    SE3 Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void SetPose(const SE3 &pose){
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    // 设置关键帧，并分配关键帧ID
    void SetKeyFrame();

    // 工厂构建模式，分配ID
    static std::shared_ptr<Frame> CreateFrame();
};

}

#endif