#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// 2nd derivative of f(x,y)
// d^2f(x,y) = d^2(f)/d(x^2) + d^2(f)/d(y^2);
// d^2(f)/d(x^2) = df(x+1,y) - df(x,y)  / (x+1 - x);
// -- df(x+1,y) = f(x+1,y) - f(x,y) / (x+1 - x) = f(x+1,y) - f(x,y);
// -- df(x, y) = f(x, y) - f(x-1, y) / (x - (x-1)) = f(x,y) - f(x-1,y);
// d^2(f)/d(x^2) = f(x+1,y) - f(x,y) - (f(x,y) - f(x-1,y)) = f(x+1,y) + f(x-1,y) - 2f(x,y);
// d^2(f)/d(y^2) = f(x, y+1) + f(x, y-1) - 2f(x,y);
// ==> d^2f(x,y) = f(x+1,y) + f(x-1,y) + f(x, y+1) + f(x, y-1) - 4f(x,y)
// 0  1  0 
// 1 -4  1
// 0  1  0
// now we get mask with 90 degree rotation no changement, we can improve it to 45 degree no changement
// 1  1  1 
// 1 -8  1
// 1  1  1
// 
// now we get 2nd derivative of each pixel
// if we want to sharpen image, we need add this with source image
// since center grid is negative, we multiply with -1 then add with source image

void laplaceFilter(cv::Mat &input, cv::Mat &output) {
    
    // init laplaceFilter mask
    int mask[3][3] = {{1, 1, 1},
                      {1,-8, 1}, 
                      {1, 1, 1}};
    
    cv::Mat pad;
    const int r = 3 / 2;  // kenerl size = 3
    cv::copyMakeBorder(input, pad, r, r, r, r, cv::BORDER_REFLECT);
    
    const int row = pad.rows - r;
    const int col = pad.cols - r;
    
    cv::Mat temp = cv::Mat::zeros(input.rows, input.cols, CV_16S);
    
    short minVal = 256, maxVal = 0;
    
    for (int i = r; i < row; ++i) {
        for (int j = r; j < col; ++j) {
            
            short val = 0;
            
            for (int a = -r; a <= r; ++a) {
                for (int b = -r; b <= r; ++b) {
                    
                    val += pad.at<uchar>(i+a, j+b) * mask[a+r][b+r];
                    
                }
            }
            
            if (val < minVal)
                minVal = val;
            else if (val > maxVal)
                maxVal = val;
            
            temp.at<short>(i-r, j-r) = static_cast<short>(val);
        }
    }

    // since temp pixel value may < 0 or > 255, we normalized to [0, 255]
    
    short gap = maxVal - minVal;
    
    for (int i = 0; i < row - r; ++i) {
        for (int j = 0; j < col - r; ++j) {
            output.at<uchar>(i, j) = static_cast<uchar>(255*(temp.at<short>(i,j) - minVal) / gap);
        }
    }
}


int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "info incomplete! usage: code_name image_path" << std::endl;
        return -1;
    }
    
    // turn input image into grayscale
    cv::Mat input = cv::imread(argv[1], 1);
    cv::imshow("input", input);
    cv::waitKey(0);
    
    // gaussian blur
    // if sigmaX = sigmaY = 0, sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    cv::GaussianBlur(input, input, cv::Size(3, 3), 0, 0);
    
    // convert to gray
    cv::Mat gray;
    cv::cvtColor(input, gray, CV_BGR2GRAY);
    cv::imshow("gray", gray);
    cv::waitKey(0);
    
    // my function
    cv::Mat output = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
    laplaceFilter(gray, output);
    cv::imshow("my function", output);
    cv::waitKey(0);
    
    cv::Mat add1;
    cv::addWeighted(gray, 1.0, output, -1.0, 0.0, add1, CV_32F);
    cv::convertScaleAbs(add1, add1);
    cv::imshow("add1", add1);
    cv::waitKey(0);
    
    // opencv function(notice: output of Laplacian was not normalized)
    cv::Mat output2, add2;
    cv::Laplacian(gray, output2, CV_16S, 3);
    cv::convertScaleAbs(output2, output2);
    cv::imshow("opencv", output2);
    cv::waitKey(0);
    
    cv::addWeighted(gray, 1.0, output2, -1.0, 0.0, add2, CV_32F);
    cv::convertScaleAbs(add2, add2);
    cv::imshow("add2", add2);
    cv::waitKey(0);
    
    return 0;
}
