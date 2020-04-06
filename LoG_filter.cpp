#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>


void LoGFilter(cv::Mat &input, cv::Mat &output, const int kernelSize, float sigma) {
    
    if (sigma == 0)
        sigma = 0.3*((kernelSize-1)*0.5 - 1) + 0.8;
    
    const int ctr = kernelSize / 2;
    
    // compute LoG mask
    cv::Mat mask = cv::Mat::zeros(kernelSize, kernelSize, CV_32FC1);
    float sigma2 = sigma * sigma;
    float sigma6 = std::pow(sigma, 6);
    float sum = 0.0f;
    
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            
            int x2 = std::pow(i - ctr, 2);
            int y2 = std::pow(j - ctr, 2);
            
            float val = (x2 + y2 - 2 * sigma2) / (2 * M_PI * sigma6) * std::exp((-x2 - y2) / (2 * sigma2));
            sum += val;
            mask.at<float>(i, j) = val;
        }
    }
    
//     // TODO: do we need normlized mask ?
//     for (int i = 0; i < kernelSize; ++i) {
//         for (int j = 0; j < kernelSize; ++j) {
//             mask.at<float>(i, j) /= sum;
//         }
//     }
    
    // padding to handle border 
    cv::Mat pad;
    const int r = kernelSize / 2;
    cv::copyMakeBorder(input, pad, r, r, r, r, cv::BORDER_REFLECT);
    const int row = pad.rows - r;
    const int col = pad.cols - r;
    
    cv::Mat temp = cv::Mat::zeros(input.rows, input.cols, CV_32FC1);
    float maxVal = -1.0f;
    float minVal = 256.0f;
    for (int i = r; i < row; ++i) {
        for (int j = r; j < col; ++j) {
            
            float cur = 0.0f;
            for (int m = -r; m <= r; ++m) {
                for (int n = -r; n <= r; ++n) {
                    cur += static_cast<float>(pad.at<uchar>(i+m, j+n)) * mask.at<float>(m+r, n+r);
                }
            }

                
            if (cur < minVal)
                minVal = cur;
            else if (cur > maxVal)
                maxVal = cur;
            
            temp.at<float>(i-r, j-r) = cur;

        }
    }

        
    float gap = maxVal - minVal;
    
    for (int i = 0; i < row - r; ++i) {
        for (int j = 0; j < col - r; ++j) {
            output.at<uchar>(i, j) = static_cast<uchar>(255 * (temp.at<float>(i,j) - minVal) / gap);
        }
    }
    
}


int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "info incomplete! usage: programe_name image_path" << std::endl;
    }
    
    cv::Mat input = cv::imread(argv[1], 1);
    cv::imshow("input", input);
    cv::waitKey(0);
    
    cv::Mat gray;
    cv::cvtColor(input, gray, CV_BGR2GRAY);
    cv::imshow("gray", gray);
    cv::waitKey(0);
    
    const float sigma = 0.8;
    const int kernelSize = 7;
    
    cv::Mat output1 = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
    LoGFilter(gray, output1, kernelSize, sigma);
    cv::imshow("my function", output1);
    cv::waitKey(0);
    
    cv::Mat add;
    cv::addWeighted(gray, 1.0, output1, -1.0, 0.0, add, CV_32F);
    cv::convertScaleAbs(add, add);
    cv::imshow("add", add);
    cv::waitKey(0);
    
    return 0;
}

