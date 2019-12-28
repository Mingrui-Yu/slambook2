#include "myslam/visual_odometry.h"
#include "chrono"
#include "myslam/config.h"
#include <opencv2/opencv.hpp>


namespace myslam{

VisualOdometry::VisualOdometry(std::string &config_path): config_file_path_(config_path) {}

bool VisualOdometry::Init(){
    // read from config file
    if (Config::SetParameterFile(config_file_path_) == false){
        return false;
    }

    dataset_ = Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    CHECK_EQ(dataset_->Init(), true);

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    viewer_ = Viewer::Ptr(new Viewer);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    viewer_->SetMap(map_);

    return true;
}

void VisualOdometry::Run(){
    while(1){
        LOG(INFO) << "VO is running";
        if(Step() == false){
            break;
        }
    }

    backend_->Stop();
    viewer_->Close();

    LOG(INFO) << "VO exit";
}


bool VisualOdometry::Step(){
    Frame::Ptr new_frame = dataset_->NextFrame();
    if(new_frame == nullptr) return false;

    auto t1 = std::chrono::steady_clock::now();

    bool success = frontend_->AddFrame(new_frame);

    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    LOG(INFO) << "VO cost time: " << time_used.count() << "seconds.";

    return success;
}

void VisualOdometry::SaveAllKeyPoseInTXT(std::string &save_file){
    std::ofstream outfile;
    outfile.open(save_file, std::ios_base::out|std::ios_base::trunc);
    std::map<unsigned long, Frame::Ptr> poses_map;

    for (auto &kf: map_->GetAllKeyFrames()){
        unsigned long keyframe_id = kf.first;
        Frame::Ptr keyframe = kf.second;
        poses_map.insert(make_pair(keyframe_id, keyframe));
    }
    
    for (auto &kf: poses_map){
        unsigned long keyframe_id = kf.first;
        Frame::Ptr keyframe = kf.second;
        SE3 frame_pose = keyframe->Pose().inverse();
        Vec3 pose_t = frame_pose.translation();
        Mat33 pose_R = frame_pose.rotationMatrix();
        Eigen::Quaterniond pose_q = Eigen::Quaterniond(pose_R);

        outfile << keyframe_id << " " << pose_t.transpose() << " " << pose_q.coeffs().transpose() << std::endl;
    }
    outfile.close();
}

void VisualOdometry::ReadAndDrawTrajectory(std::string &trajectory_file){
    // 读取存有all key poses的text文件
    std::ifstream fin(trajectory_file);
    if(!fin){
        LOG(ERROR) << "cannot find trajectory file at " << trajectory_file;
    }

    std::vector<SE3> poses;
    // 依次读取数据
    while (!fin.eof()) {
        unsigned long id;
        double tx, ty, tz, qx, qy, qz, qw;
        fin >> id >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Eigen::Quaterniond q = Eigen::Quaterniond(qw, qx, qy, qz);
        q.normalize();
        SE3 Twr(q, Vec3(tx, ty, tz));
        poses.push_back(Twr);
    }
    viewer_->DrawTrajectory(poses);
}



} // namespace myslam