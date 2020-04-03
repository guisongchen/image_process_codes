#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "info incomplete! usage: programe_name image_path" << std::endl;
    }
    
    // -1 is original, 0 is gray 1 is BGR
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::imshow("input", input);
    cv::waitKey(0);
    
    //-------channel switch-----------
    /*
    // use opencv function
    cv::Mat output1;
    cv::cvtColor(input, output1, CV_BGR2RGB);
    cv::imshow("output1", output1);
    cv::waitKey(0);
    
    // use channel split and merge
    
    cv::Mat output2;
    std::vector<cv::Mat> chs;
    cv::split(input, chs);
    std::swap(chs[0], chs[2]);
    cv::merge(chs, output2);
    
    cv::imshow("output2", output2);
    cv::waitKey(0);
    */
    
    //-------grayscale-----------
    
    // use opencv function
    cv::Mat gray1;
    cv::cvtColor(input, gray1, CV_BGR2GRAY);
    cv::imshow("gray1", gray1);
    cv::waitKey(0);
    
    // we get every gray value of each pixel, there are different ways
    // 1. gray[i][j] = R[i][j] or G[i][j] or B[i][j]
    // 2. gray[i][j] = max(R[i][j], G[i][j], B[i][j])
    // 3. gray[i][j] = average(R[i][j], G[i][j], B[i][j])
    // 4. gray[i][j] = 0.072*B[i][j] + 0.715*G[i][j] + 0.213*R[i][j] <--opencv
    // 5. gray[i][j] = 0.11*B[i][j] + 0.59*G[i][j] + 0.3*R[i][j]  <-- human eye, YUV model
    

    int row = input.rows;
    int col = input.cols;
    
    cv::Mat gray2 = cv::Mat::zeros(row, col, CV_8UC1); // one channel
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            gray2.at<uchar>(i,j) = 0.072 * input.at<cv::Vec3b>(i,j)[0] +
                                   0.715 * input.at<cv::Vec3b>(i,j)[1] +
                                   0.213 * input.at<cv::Vec3b>(i,j)[2];
        }
    }
    
    cv::imshow("gray2", gray2);
    cv::waitKey(0);
    
    cv::Mat gray3 = cv::Mat::zeros(row, col, CV_8UC1); // one channel
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            gray3.at<uchar>(i,j) = 0.11 * input.at<cv::Vec3b>(i,j)[0] +
                                   0.59 * input.at<cv::Vec3b>(i,j)[1] +
                                   0.3 * input.at<cv::Vec3b>(i,j)[2];
        }
    }
    
    cv::imshow("gray3", gray3);
    cv::waitKey(0);

    return 0;
}
