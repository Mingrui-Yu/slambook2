#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <boost/format.hpp> // for formatting strings
 
using namespace cv;
using namespace std;

int main(int argc, char **argv) 
{
    ifstream fin("calibdata.txt"); /* 标定所用图像文件的路径 */
	ofstream fout("caliberation_result.txt");  /* 保存标定结果的文件 */	
    //读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化	
	int image_count=0;  /* 图像数量 */
	int valid_image_count = 0; // 能成功提取出角点的图像数量
	Size image_size;  /* 图像的尺寸 */
	Size board_size = Size(9,6);    /* 标定板上每行、列的角点数 */
	vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f>> image_points_seq; // 保存检测到的所有角点
	string filename;
	int count= -1 ;//用于存储角点个数。

	
    while (getline(fin, filename))
    {
		image_count++;
        cv::Mat rawImageInput = cv::imread(filename);
        cv::Mat imageInput = rawImageInput;
        if (image_count == 1)
        {
            image_size.width = imageInput.cols;
            image_size.height = imageInput.rows;
            cout << "image_size = " <<  image_size << endl;
			cout << "开始提取角点………………" << endl;
        }

        // 用于观察检验输出
		cout<<"image_count = "<<image_count<<endl;	

        boost::format fmt("%s %d");
        // cv::imshow((fmt % "imageInput" % image_count).str(), imageInput);
        // cv::waitKey(0);
        
        if (0 == findChessboardCorners(imageInput,board_size,image_points_buf))
		{			
			cout<<"can not find chessboard corners!\n"; //找不到角点
			// exit(1);
		} 
		else 
		{
			valid_image_count++;
			Mat view_gray;
			cvtColor(imageInput,view_gray,CV_RGB2GRAY);
			/* 亚像素精确化 */
			//find4QuadCornerSubpix(view_gray,image_points_buf,Size(5,5)); //对粗提取的角点进行精确化
			cornerSubPix(view_gray,image_points_buf,Size(11,11),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));
			
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点
			/* 在图像上显示角点位置 */
			drawChessboardCorners(view_gray,board_size,image_points_buf,false); //用于在图片中标记角点
			cv::imshow((fmt % "Camera Calibration" % image_count).str(),view_gray);//显示图片	
		}
    }
    cv::waitKey(500);
	cout << "角点提取完成，" << "有效图片数=" << valid_image_count << endl << endl;


	// 摄像机标定
	cout << "开始标定......" << endl;
	Size square_size = Size(50, 50); // 实际测量得到的标定板上每个棋盘格的大小
	vector<vector<Point3f>> objectPoints;  // 保存标定板上角点的三维坐标

	// vector<vector<Point3f>> objectPoints(1);
    // calcBoardCornerPositions(board_size, square_size, objectPoints[0]);

    // objectPoints.resize(image_points_seq.size(), objectPoints[0]);

	Mat cameraMatrix= Mat::eye(3, 3, CV_64F); // 定义相机内参矩阵
	vector<int> point_counts;  // 每幅图像中角点的数量
	Mat distCoeffs = Mat::zeros(1, 5, CV_64F);  // 摄像机的五个畸变系数: k1 k2 p1 p2 k2
	vector<Mat> tvecsMat;  //每幅图像的评议向量  
	vector<Mat> rvecsMat;  //每幅图像的旋转向量
	// 计算角点的真实坐标（假定在z=0平面上）
	for (int n=0; n < valid_image_count; n++)
	{
		vector<Point3f> tempPointSet;
		for (int u=0; u < board_size.height; ++u)
		{
			for (int v=0; v < board_size.width; ++v)
			{
				Point3f realPoint;
				realPoint.x = v * square_size.width;
				realPoint.y = u * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		// cout << "真实角点：" << endl << tempPointSet << endl;
		objectPoints.push_back(tempPointSet);
	}
	// 标定
	int flags = 0;
	flags = CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_FIX_ASPECT_RATIO;
	calibrateCamera(objectPoints, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, flags);
	cout << "标定完成!" << endl << endl;


	// 对标定结果进行评价
	cout << "开始评价标定结果......" << endl;
	vector<Point2f> image_points_new;

	for(int i=0; i < image_count; i++)
	{
		vector<Point3f> tempPointSet = objectPoints[i];
		projectPoints(objectPoints[i], rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points_new);

	}
	
    




	




    return 0;
}