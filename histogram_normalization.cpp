#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

cv::Mat drawHist(cv::Mat &input) {
    
    std::vector<cv::Mat> channels;
    cv::split(input, channels); // BGR
    
    const int histSize = 255;
    float range[] = {0, 255};
    const float* histRange = range;
    
    cv::Mat histR, histG, histB;
    const bool uniform = true;
    const bool accumulate = false;
    
    cv::calcHist(&channels[0], 1, 0, cv::Mat(), histB, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&channels[1], 1, 0, cv::Mat(), histG, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&channels[2], 1, 0, cv::Mat(), histR, 1, &histSize, &histRange, uniform, accumulate);
    
    const int width = 500, height = 500;
    const int binWidth = cvRound(static_cast<double>(width) / histSize);
    
    cv::Mat histImg(height, width, CV_8UC3, cvScalar(0,0,0));
    
    // normlized to [0, histImg.rows]
    cv::normalize(histB, histB, 0, histImg.rows, cv::NORM_MINMAX);
    cv::normalize(histG, histG, 0, histImg.rows, cv::NORM_MINMAX);
    cv::normalize(histR, histR, 0, histImg.rows, cv::NORM_MINMAX);
    
    for (int i = 1; i < histSize; ++i) {
        cv::line(histImg, cv::Point(binWidth*(i-1), height - cvRound(histB.at<float>(i-1))),
                          cv::Point(binWidth*(i), height - cvRound(histB.at<float>(i))),
                          cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImg, cv::Point(binWidth*(i-1), height - cvRound(histG.at<float>(i-1))),
                          cv::Point(binWidth*(i), height - cvRound(histG.at<float>(i))),
                          cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(histImg, cv::Point(binWidth*(i-1), height - cvRound(histR.at<float>(i-1))),
                          cv::Point(binWidth*(i), height - cvRound(histR.at<float>(i))),
                          cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    
    return histImg;
}

void histNormalization(cv::Mat &input, const int lower, const int upper) {
    
    const int row = input.rows;
    const int col = input.cols;
    const int ch = input.channels();
    
    // get min and max of input image
    int minVal = 256, maxVal = -1;
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            for (int k = 0; k < ch; ++k) {
                int val = input.at<cv::Vec3b>(i,j)[k];
                minVal = std::min(val, minVal);
                maxVal = std::max(val, maxVal);
            }
        }
    }
    
    // normalization
    const float rate = (upper-lower) / (maxVal-minVal);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            for (int k = 0; k < ch; ++k) {
                int val = input.at<cv::Vec3b>(i,j)[k];
                
                if (val < minVal) {
                    input.at<cv::Vec3b>(i,j)[k] = lower;
                } else if (val > maxVal) {
                    input.at<cv::Vec3b>(i,j)[k] = upper;
                } else {
                    input.at<cv::Vec3b>(i,j)[k] = static_cast<uchar>(rate*(val - minVal) + lower);
                }
            }
        }
    }
    
    std::cout << "before: (" << minVal << "," << maxVal << ")" << std::endl;
    std::cout << "after: (" << lower << "," << upper << ")" << std::endl;
}

int main (int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "info incomplete! usage: programe_name image_path" << std::endl;
    }
    
    cv::Mat input = cv::imread(argv[1], 1);
    cv::imshow("input", input);
    cv::waitKey(0);
    
    cv::Mat histBefore = drawHist(input);
    cv::imshow("histBefore", histBefore);
    cv::waitKey(0);
    
    // normalization
    const int lower = 0, upper = 255;
    cv::Mat output = input.clone();
    histNormalization(output, lower, upper);
    cv::imshow("output", output);
    cv::waitKey(0);
    
    cv::Mat histAfter = drawHist(output);
    cv::imshow("histAfter", histAfter);
    cv::waitKey(0);
    
    return 0;
}

