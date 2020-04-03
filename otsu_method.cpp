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
    
    
    //-------grayscale-----------
    
    // use opencv function
    cv::Mat gray1;
    cv::cvtColor(input, gray1, CV_BGR2GRAY);
    cv::imshow("gray1", gray1);
    cv::waitKey(0);
    
    
    int row = gray1.rows;
    int col = gray1.cols;
    
    //-------binarize-----------
    // 1. use given threshold
    
    int th = 128;
    
    // notice: '=' means shallow copy, only get reference
    // if we don't want to change gray1, we use deep copy(clone())
    cv::Mat bin1 = gray1.clone(); 
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            if (bin1.at<uchar>(i,j) < th) {
                bin1.at<uchar>(i,j) = 0;
            } else {
                bin1.at<uchar>(i,j) = 255;
            }
        }
    }
    
    std::cout << "threshold value by given: " << th << std::endl;
    
    cv::imshow("bin1", bin1);
    cv::waitKey(0);
    
    // 2. use otsu_method, aim to find best threshold to distinguish background and foreground
    // more details see wikipedia
    // --> find minmun within class varience(sum of background and foreground varience)
    // --> find maximum of between class varience
    // choose method 2 for convience
    
    // step 1: var_between = var - var_within = wb(ub - u)^2 + wf(uf - u)^2
    //         u = wb*ub + wf*uf
    
    float bestTh = 0.0f, maxVar = 0.0f;
    
    // iterate to find best threshold value
    for (int t = 0; t < 256; ++t) {
        
        // init for each candidate t
        float w0 = 0.0f, w1 = 0.0f, m0 = 0.0f, m1 = 0.0f;
        
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
            
                int cur = gray1.at<uchar>(i,j);
                
                // count pixel num and compute sum for background or foreground 
                if (cur < t) {
                    ++w0;
                    m0 += cur;
                } else {
                    ++w1;
                    m1 += cur;
                }
            }
        }
        
        // get average pixel value
        m0 /= w0;
        m1 /= w1;
        
        // get weight
        w0 /= row * col;
        w1 /= row * col;
        
        // compute var_between
        float var = w0 * w1 * (m0-m1) * (m0-m1);
        
        // check if is best
        if (var > maxVar) {
            maxVar = var;
            bestTh = t;
        }
    }
    
    std::cout << "threshold value by C++: " << bestTh << std::endl;
    
    // binarize use bestTh
    cv::Mat bin2 = gray1.clone();
    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            if (bin2.at<uchar>(i,j) < bestTh) {
                bin2.at<uchar>(i,j) = 0;
            } else {
                bin2.at<uchar>(i,j) = 255;
            }
        }
    }
    
    cv::imshow("bin2", bin2);
    cv::waitKey(0);
    
    // 3. use opencv function
    cv::Mat bin3 = gray1.clone();
    double threshould = cv::threshold(gray1, bin3, 0, 255, CV_THRESH_OTSU);
    
    std::cout << "threshold value by OpenCV: " << threshould << std::endl;
    
    cv::imshow("bin3", bin3);
    cv::waitKey(0);
    
    return 0;
}
