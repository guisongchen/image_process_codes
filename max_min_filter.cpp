#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


// image composed by detail and edge
// for detail area, pixel value may have less gap, max - min is close to 0
// for egde area, pixel value have big gap, max - min is much bigger than 0
// using this we can tell the differece between detail and edge

// notice: when apply filter, we must leave input or pad image untouched(not update value)
//         we need original value during next iterations, so update computed value to another image(e.g. output)

void maxMinFilter(cv::Mat &input, cv::Mat &output, int kernelSize) {
    
    // padding to handle borders
    cv::Mat pad;
    const int r = kernelSize / 2;
    cv::copyMakeBorder(input, pad, r, r, r, r, cv::BORDER_REFLECT);
    
    // apply maxMinFilter
    const int row = pad.rows - r;
    const int col = pad.cols - r;
    
    for (int i = r; i < row; ++i) {
        for (int j = r; j < col; ++j) {
            
            int maxVal = -1;
            int minVal = 256;
            
            // search for max and min
            for (int m = -r; m <= r; ++m) {
                for (int n = -r; n <= r; ++n) {
                    
                    int cur = static_cast<int>(pad.at<uchar>(i+m, j+n));
                    
                    if ( cur > maxVal) {
                        maxVal = cur;
                    } 
                    
                    if (cur < minVal) {
                        minVal = cur;
                    }
                }
            }
            
            // max - min
            output.at<uchar>(i-r, j-r) = static_cast<uchar>(maxVal -minVal);
        }
    }
}


int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "info incomplete! usage: program_name image_path" << std::endl;
        return -1;
    }
    
    // turn input image into gray
    cv::Mat gray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::imshow("gray", gray);
    cv::waitKey(0);
    
    // apply maxMinFilter to detect edge
    cv::Mat output = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
    maxMinFilter(gray, output, 3);
    cv::imshow("output", output);
    cv::waitKey(0);
    
    return 0;
}
