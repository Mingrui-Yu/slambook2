#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{
    cv::Mat image;
    image = cv::imread("ubuntu.png");

    if (image.data == nullptr)
    {
        cerr << "图片不存在" << endl;
        return 0;
    }

    cout << "width = " << image.cols << ", height = " << image.rows << endl;

    // cv::imshow("image", image);
    // cv::waitKey(0);

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    for (size_t y=0; y < image.rows; y++)
    {
        unsigned char *row_ptr = image.ptr<unsigned char>(y);
        for (size_t x=0; x < image.cols; x++)
        {
            unsigned char *data_ptr = &row_ptr[x * image.channels()];

            for (int c=0; c!=image.channels(); c++)
            {
                unsigned char data = data_ptr[c];
            }
        }

    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2-t1);
    cout << "time used for traversal = " << time_used.count() << "s" << endl;

    cv::Mat image_another = image;
    image_another(cv::Rect(0,0,100,100)).setTo(0);
    cv::imshow("image (after image_another)", image);

    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0,0,100,100)).setTo(255);
    cv::imshow("image (after image_clone)", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);


    cv::destroyAllWindows();

    return 0;
}