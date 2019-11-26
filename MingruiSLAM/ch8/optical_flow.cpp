#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;


string file_1 = "./LK1.png";  // first image
string file_2 = "./LK2.png";  // second image




class OpticalFlowTracker{
public:
    OpticalFlowTracker(const Mat &img1_,
                       const Mat &img2_,
                       const vector<KeyPoint> &kp1_,
                       vector<KeyPoint> &kp2_,
                       vector<bool> &success_,
                       bool inverse_ = true,
                       bool has_initial_ = false):
                       img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
                       has_initial(has_initial_) {}
    
    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);


void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false, bool has_initial = false);





inline float GetPixelValue(const Mat &img, double x, double y){
    if (x < 0) x = 0;
    else if (x > img.cols - 1) x = img.cols - 1;
    if (y < 0) y = 0;
    else if (y > img.rows - 1) y = img.rows - 1;

    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    float pixelValue = (1-xx)*(1-yy)*data[0] + (xx)*(1-yy)*data[1] 
                        + (1-xx)*(yy)*data[img.step] + (xx)*(yy)*data[img.step+1];
    return pixelValue;
}







int main(int argc, char **argv){
    cv::Mat img_1 = cv::imread(file_1, 0);
    cv::Mat img_2 = cv::imread(file_2, 0);

    // 从img1中提取特征点，使用GFTT特征
    vector<KeyPoint> kp_1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
    detector->detect(img_1, kp_1);


    // OpenCV LK光流
    vector<Point2f> pts_1, pts_2;
    for (auto &kp:kp_1){
        pts_1.push_back(kp.pt);
    }
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img_1, img_2, pts_1, pts_2, status, error);
    // 画图：By OpenCV LK光流
    Mat img_2_CV;
    cv::cvtColor(img_2,img_2_CV, CV_GRAY2BGR);
    for (int i=0; i<pts_2.size(); i++){
        if (status[i]){
            cv::circle(img_2_CV, pts_2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img_2_CV, pts_1[i], pts_2[i], cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("By OpenCV", img_2_CV);


    // 单层LK光流
    vector<KeyPoint> kp_2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img_1, img_2, kp_1, kp_2_single, success_single);
    // 画图：单层LK光流
    Mat img_2_single;
    cv::cvtColor(img_2,img_2_single, CV_GRAY2BGR);
    for (int i=0; i<kp_2_single.size(); i++){
        if (success_single[i]){
            cv::circle(img_2_single, kp_2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img_2_single, kp_1[i].pt, kp_2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("By single", img_2_single);

    // 反向单层LK光流
    vector<KeyPoint> kp_2_single_inverse;
    vector<bool> success_single_inverse;
    OpticalFlowSingleLevel(img_1, img_2, kp_1, kp_2_single_inverse, success_single_inverse, true);
    // 画图：单层LK光流
    Mat img_2_single_inverse;
    cv::cvtColor(img_2,img_2_single_inverse, CV_GRAY2BGR);
    for (int i=0; i<kp_2_single_inverse.size(); i++){
        if (success_single_inverse[i]){
            cv::circle(img_2_single_inverse, kp_2_single_inverse[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img_2_single_inverse, kp_1[i].pt, kp_2_single_inverse[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("By single_inverse", img_2_single_inverse);


    // 多层LK光流
    vector<KeyPoint> kp_2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(img_1, img_2, kp_1, kp_2_multi, success_multi, true);
    // OpticalFLowMultiLevel(img_1, img_2, kp_1, kp_2_multi, success_multi, true);
    // 画图：多层LK光流
    Mat img_2_multi;
    cv::cvtColor(img_2,img_2_multi, CV_GRAY2BGR);
    for (int i=0; i<kp_2_multi.size(); i++){
        if (success_multi[i]){
            cv::circle(img_2_multi, kp_2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img_2_multi, kp_1[i].pt, kp_2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("By multi", img_2_multi);


    








    cv::waitKey(0);
    return 0;
}








void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_initial){
    
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(Range(0, kp1.size()), 
        std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}


void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) {

    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    vector<Mat> pyr1, pyr2;
    for (int i = 0; i < pyramids; i++){
        if(i==0){
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else{
            Mat img1_ptr, img2_ptr;
            cv::resize(pyr1[i-1], img1_ptr, Size(), pyramid_scale, pyramid_scale);
            cv::resize(pyr2[i-1], img2_ptr, Size(), pyramid_scale, pyramid_scale);
            pyr1.push_back(img1_ptr);
            pyr2.push_back(img2_ptr);
        }
    }

    vector<KeyPoint> kp1_ptr, kp2_ptr;
    for (auto &kp: kp1){
        auto kp_top = kp;
        kp_top.pt *= (pow(pyramid_scale, pyramids - 1));
        kp1_ptr.push_back(kp_top);
        kp2_ptr.push_back(kp_top);
    }

    for(int level = pyramids - 1; level >= 0; level--){
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_ptr, kp2_ptr, success, inverse, true);

        if (level > 0){
            for (auto &kp:kp1_ptr){
                kp.pt /= pyramid_scale; 
            }
            for (auto &kp:kp2_ptr){
                kp.pt /= pyramid_scale; 
            }
        }
    }

    for(auto &kp: kp2_ptr){
        kp2.push_back(kp);
    }
}








// void OpticalFlowTracker::calculateOpticalFlow(const Range &range){
//     int half_patch_size = 4;
//     int iterations = 10;
//     for(size_t i = range.start; i < range.end; i++){
//         auto kp = kp1[i];
//         double dx = 0, dy = 0;
//         if (has_initial){
//             dx = kp2[i].pt.x - kp.pt.x;
//             dy = kp2[i].pt.y - kp.pt.y;
//         }

//         double cost = 0, lastCost = 0;
//         bool succ = true;

//         // Gauss-Newton iterations
//         Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
//         Eigen::Vector2d b = Eigen::Vector2d::Zero();
//         Eigen::Vector2d J;

//         for (int iter = 0; iter < iterations; iter++){

//             cost = 0;
//             if (inverse == false){
//                 H = Eigen::Matrix2d::Zero();
//                 b = Eigen::Vector2d::Zero();
//             }
//             else{
//                 b = Eigen::Vector2d::Zero();
//             }

//             for (int x = -half_patch_size; x < half_patch_size; x++){
//                 for (int y = -half_patch_size; y < half_patch_size; y++) {
//                     double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
//                                    GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian
//                     if (inverse == false) {
//                         J = -1.0 * Eigen::Vector2d(
//                             0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
//                                    GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
//                             0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
//                                    GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
//                         );
//                     } else if (iter == 0) {
//                         // in inverse mode, J keeps same for all iterations
//                         // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
//                         J = -1.0 * Eigen::Vector2d(
//                             0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
//                                    GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
//                             0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
//                                    GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
//                         );
//                     }
//                     // compute H, b and set cost;
//                     b += -error * J;
//                     cost += error * error;
//                     if (inverse == false || iter == 0) {
//                         // also update H
//                         H += J * J.transpose();
//                     }
//                 }
//             }


//             Eigen::Vector2d update = H.ldlt().solve(b);

//             if (std::isnan(update[0])) {
//                 // sometimes occurred when we have a black or white patch and H is irreversible
//                 cout << "update is nan" << endl;
//                 succ = false;
//                 break;
//             }

//             if (iter > 0 && cost > lastCost) {
//                 break;
//             }

//             // update dx, dy
//             dx += update[0];
//             dy += update[1];
//             lastCost = cost;
//             succ = true;

//             if (update.norm() < 1e-2) {
//                 // converge
//                 break;
//             }
//         }

//         success[i] = succ;

//         // set kp2
//         kp2[i].pt = kp.pt + Point2f(dx, dy);
//     }
// }


void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias
        Eigen::Vector2d J;  // jacobian
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // converge
                break;
            }
        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}