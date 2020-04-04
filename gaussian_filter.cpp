#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

// we can compute 2d gaussian or 1d gaussian
cv::Mat computeMask(float sigma, int r, int dim) {
    
    // compute kernel matrix(mask)
    const int ctr = r / 2;
    const float sigma2 = sigma * sigma;
    float x2 = 0.0f, y2 = 0.0f, sum = 0.0f;
    
    if (dim == 1) {
        
        cv::Mat mask = cv::Mat::zeros(r, 1, CV_32FC1);
        
        for (int i = 0; i < r; ++i) {
            x2 = std::pow(i-ctr, 2);
            float p = std::exp(-x2 / (2*sigma2));
            sum += p;
            mask.at<float>(i,0) = p;
        }
        
        for (int i = 0; i < r; ++i) {
            mask.at<float>(i,0) /= sum;
        }
        
        return mask;
    }
    
    
    else if (dim == 2) {
    
        cv::Mat mask = cv::Mat::zeros(r, r, CV_32FC1);
        
        // get possible value of each grid
        for (int i = 0; i < r; ++i) {
            
            x2 = std::pow(i-ctr, 2);
            
            for (int j = 0; j < r; ++j) {
                
                y2 = std::pow(j-ctr, 2);
                
                float p = std::exp(-(x2 + y2) / (2 * sigma2));
                sum += p;
                mask.at<float>(i,j) = p;
            }
        }
        
        // divide sum to make all grid added equal 1
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < r; ++j) {
                mask.at<float>(i,j) /= sum ;
            }
        }
        
        
        return mask;
    }
    
    
    return cv::Mat();
    
}

void gaussianFilter(cv::Mat &input, cv::Mat &output, cv::Mat &mask) {
    
    const int ch = input.channels();
    const int r = mask.rows / 2;
    
    // padding to handle border 
    cv::Mat pad;
    cv::copyMakeBorder(input, pad, r, r, r, r, cv::BORDER_REFLECT);
    const int row = pad.rows - r;
    const int col = pad.cols - r;
    
    for (int i = r; i < row; ++i) {
        for (int j = r; j < col; ++j) {
            
            float sum[3] = {0.0f};
            
            for (int m = -r; m <= r; ++m) {
                for (int n = -r; n <= r; ++n) {
                    
                    // make sure input is 1 channel or 3 channels
                    // notice mask datatype is float(we define in computeMask)
                    if (ch == 1) {
                        sum[0] += mask.at<float>(r+m, r+n) * pad.at<uchar>(i+m, j+n); 
                    } else if (ch == 3) {
                        cv::Vec3b bgr = pad.at<cv::Vec3b>(i+m, j+n);
                        float p = mask.at<float>(r+m, r+n);
                        sum[0] += bgr[0] * p;
                        sum[1] += bgr[1] * p;
                        sum[2] += bgr[2] * p;
                    }
                }
            }
            
            // constrain pixel value between 0 and 255
            for (int k = 0; k < ch; ++k) {
                if (sum[k] < 0) {
                    sum[k] = 0;
                } else if (sum[k] > 255) {
                    sum[k] = 255;
                }
            }
            
            // set value
            // notice we can't change value of pad, we need keep it to do next filter iteration
            // we save new value to output image
            if (ch == 1) {
                output.at<uchar>(i-r, j-r) = static_cast<uchar>(sum[0]);
            } else if (ch == 3) {
                cv::Vec3b bgr = {static_cast<uchar>(sum[0]), 
                                 static_cast<uchar>(sum[1]), 
                                 static_cast<uchar>(sum[2])};
                                 
                output.at<cv::Vec3b>(i-r, j-r) = bgr;
            }
        }
    }
}


// g(x, y) = g(x) * g(y)
// we split g(x,y) into two gaussian filters g(x) and g(y)
// apply g(x) to x direction of source image, apply g(y) to y direction
// then add them up, we get finnal result, time complicity: O(n^2) -> O(n)

void splitGaussianFilter(cv::Mat &input, cv::Mat &output, cv::Mat &mask) {
    
    cv::Mat pad;
    const int r = mask.rows / 2;
    cv::copyMakeBorder(input, pad, r, r, r, r, cv::BORDER_REFLECT);
    const int row = pad.rows - r;
    const int col = pad.cols - r;
    const int ch = input.channels();
    
    
    // do x-direction convolution
    cv::Mat imgX = pad.clone();
    for (int i = r; i < row; ++i) {
        for (int j = r; j < col; ++j) {
            
            float sum[3] = {0.0f};

            for (int n = -r; n <= r; ++n) {
                    
                // make sure input is 1 channel or 3 channels
                // notice mask datatype is float(we define in computeMask)
                if (ch == 1) {
                    sum[0] += mask.at<float>(r+n, 0) * pad.at<uchar>(i, j+n); 
                } else if (ch == 3) {
                    cv::Vec3b bgr = pad.at<cv::Vec3b>(i, j+n);
                    float p = mask.at<float>(r+n, 0);
                    sum[0] += bgr[0] * p;
                    sum[1] += bgr[1] * p;
                    sum[2] += bgr[2] * p;
                }
            }
            
            // set value
            // notice we can't change value of pad, we need keep it to do next filter iteration
            // we save new value to imgX
            if (ch == 1) {
                imgX.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            } else if (ch == 3) {
                cv::Vec3b bgr = {static_cast<uchar>(sum[0]), 
                                 static_cast<uchar>(sum[1]), 
                                 static_cast<uchar>(sum[2])};
                                 
                imgX.at<cv::Vec3b>(i, j) = bgr;
            }
        }
    }
    
    
    // do y-direction convolution
    for (int i = r; i < row; ++i) {
        for (int j = r; j < col; ++j) {
            
            float sum[3] = {0.0f};
            
            for (int m = -r; m <= r; ++m) {
                    
                // make sure input is 1 channel or 3 channels
                // notice mask datatype is float(we define in computeMask)
                if (ch == 1) {
                    sum[0] += mask.at<float>(r+m, 0) * imgX.at<uchar>(i+m, j); 
                } else if (ch == 3) {
                    cv::Vec3b bgr = imgX.at<cv::Vec3b>(i+m, j);
                    float p = mask.at<float>(r+m, 0);
                    sum[0] += bgr[0] * p;
                    sum[1] += bgr[1] * p;
                    sum[2] += bgr[2] * p;
                }
            }
            
            // constrain pixel value between 0 and 255
            for (int k = 0; k < ch; ++k) {
                if (sum[k] < 0) {
                    sum[k] = 0;
                } else if (sum[k] > 255) {
                    sum[k] = 255;
                }
            }
            
            // set value
            // notice we can't change value of imgX, we need keep it to do next filter iteration
            // we save new value to output image
            if (ch == 1) {
                output.at<uchar>(i-r, j-r) = static_cast<uchar>(sum[0]);
            } else if (ch == 3) {
                cv::Vec3b bgr = {static_cast<uchar>(sum[0]), 
                                 static_cast<uchar>(sum[1]), 
                                 static_cast<uchar>(sum[2])};
                                 
                output.at<cv::Vec3b>(i-r, j-r) = bgr;
            }
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
    
    const float sigma = 0.8;
    const int kernelSize = 3;
    
    // try gaussianFilter
    cv::Mat output1 = input.clone();
    cv::Mat mask2d = computeMask(sigma, kernelSize, 2);
    std::cout << "2d mask:\n" << mask2d << std::endl;
    gaussianFilter(input, output1, mask2d);
    
    cv::imshow("output1", output1);
    cv::waitKey(0);
    
    // try opencv function
    cv::Mat output2 = input.clone();;
    cv::GaussianBlur(input, output2, cv::Size(kernelSize,kernelSize), sigma, 0, cv::BORDER_REFLECT);
    
    cv::Mat diff1;
    cv::absdiff(output1, output2, diff1); // output1 - output2
    cv::imshow("diff1", diff1);
    cv::waitKey(0);
    
    // try split-gaussianFilter
    cv::Mat output3 = input.clone();;
    cv::Mat mask1d = computeMask(sigma, kernelSize, 1);
    std::cout << "1d mask:\n" << mask1d << std::endl;
    splitGaussianFilter(input, output3, mask1d);
    
    cv::Mat diff2;
    cv::absdiff(output3, output2, diff2); // output3 - output2
    cv::imshow("diff2", diff2);
    cv::waitKey(0);
    
    return 0;
}
