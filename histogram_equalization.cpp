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


// out = (pixelMax / pixelSum) * count_of_(pixelVal < curVal)
void histEqualization(cv::Mat &input) {
    
    const int row = input.rows;
    const int col = input.cols;
    const int ch = input.channels();
    const float rate = 255.0f / (row * col * ch);
    
    int hist[256] = {0}; 
    
    // get hist info
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            for (int k = 0; k < ch; ++k) {
                int val = input.at<cv::Vec3b>(i,j)[k];
                ++hist[val];
            }
        }
    }
    
    // equalization
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            for (int k = 0; k < ch; ++k) {
                
                int val = input.at<cv::Vec3b>(i,j)[k];
                int histSum = 0;
                
                for (int t = 0; t <= val; ++t) {
                    histSum += hist[t];
                }
                
                // set value
                input.at<cv::Vec3b>(i,j)[k] = static_cast<uchar>(histSum * rate);
            }
        }
    }
    
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
    cv::Mat output = input.clone();
    histEqualization(output);
    cv::imshow("output", output);
    cv::waitKey(0);
    
    cv::Mat histAfter = drawHist(output);
    cv::imshow("histAfter", histAfter);
    cv::waitKey(0);
    
    return 0;
}

