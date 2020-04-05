#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void sobleFilter(cv::Mat &input, cv::Mat &output, bool directionX, bool directionY) {
    
    // init soble matrix
    int gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2}, 
                    {-1, 0, 1}};
    
    int gy[3][3] = {{-1, -2, -1},
                    { 0,  0,  0}, 
                    { 1,  2,  1}};
                    
    // padding to handle border
    const int r = 3 / 2; // kernel size = 3
    cv::Mat pad;
    cv::copyMakeBorder(input, pad, r, r, r, r, cv::BORDER_REFLECT);
    
    const int row = pad.rows - r;
    const int col = pad.cols - r;
    
    for (int i = r; i < row; ++i) {
        for (int j = r; j < col; ++j) {
            
            int dx = 0, dy = 0;
            
            for (int a = -r; a <= r; ++a) {
                for (int b = -r; b <= r; ++b) {
                    
                    int cur = static_cast<int>(pad.at<uchar>(i+a, j+b));
                    
                    if (directionX)
                        dx += cur * gx[a+r][b+r];
                    
                    if (directionY)
                        dy += cur * gy[a+r][b+r];
                    
                }
            }
            
            int val = std::sqrt(dx*dx + dy*dy);
            val = std::min(255, val);
            
            output.at<uchar>(i-r, j-r) = static_cast<uchar>(val);
            
        }
    }
    
}


int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "info incomplete! usage: code_name image_path" << std::endl;
        return -1;
    }
    
    // turn input image into grayscale
    cv::Mat gray = cv::imread(argv[1], 0);
    cv::imshow("gray", gray);
    cv::waitKey(0);
    
    cv::Mat output = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
    sobleFilter(gray, output, true, true);
    cv::imshow("output", output);
    cv::waitKey(0);
    
    return 0;
}
