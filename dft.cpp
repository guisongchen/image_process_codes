#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <complex>
#include <vector>

std::vector<std::vector<std::complex<double>>> computeFourier(cv::Mat gray) {
    
    std::complex<double> val;
    const int heigh = gray.rows;
    const int width = gray.cols;
    std::vector<std::vector<std::complex<double>>> f(heigh, std::vector<std::complex<double>>(width));
    
    std::cout << "row:" << heigh << " col:" << width << std::endl;
    
    for (int i = 0; i < heigh; ++i) {
        for (int j = 0; j < width; ++j) {
            val.real(0);
            val.imag(0);
            
            for (int m = 0; m < heigh; ++m) {
                for(int n = 0; n < width; ++n) {
                    
                    // f(x,y)
                    double pixel = static_cast<double>(gray.at<uchar>(m, n));
                    
                    // -2*pi*(ux/M + vy/N)
                    double theta = -2 * M_PI * (static_cast<double>(i * m) / static_cast<double>(heigh) + 
                                                static_cast<double>(j * n) / static_cast<double>(width));
                    
                    // accumulate from 0 to M-1, 0 to N-1
                    val += std::complex<double>(std::cos(theta), std::sin(theta)) * pixel;
                }
            }
            
            // divide sqrt(M*N) to get symmetric form(both for dft and idft)
            val /= sqrt(heigh * width);
            f[i][j] = val;

        }
    }
    
    std::cout << "fourier transform is done.." << std::endl;
    
    return f;
    
}

void computeMagnitude(std::vector<std::vector<std::complex<double>>> f, cv::Mat mag) {
    const int row = f.size();
    const int col = f[0].size();
    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            std::complex<double> &cur = f[i][j];
            mag.at<float>(i,j) = std::sqrt(cur.real()*cur.real() + cur.imag()*cur.imag());
        }
    }
    
}

int main(int argc, char** argv) {
    
    if (argc < 2) {
        std::cerr << "info incomplete! usage: programe_name image_path" << std::endl;
    }
    
    cv::Mat src = cv::imread(argv[1], 1);
    cv::imshow("src", src);
    cv::waitKey(0);
    
    cv::Mat gray;
    cv::cvtColor(src, gray, CV_BGR2GRAY);
    cv::imshow("gray", gray);
    cv::waitKey(0);
    
//     // resize to reduce compute complexity
//     cv::resize(gray, gray, cv::Size(0, 0), 0.5, 0.5, CV_INTER_NN);
//     cv::imshow("gray", gray);
//     cv::waitKey(0);
    
    // compute F(u,v)
    std::vector<std::vector<std::complex<double>>> f = computeFourier(gray);
    
    // compute magnitude of F(u,v)
    cv::Mat mag = cv::Mat(gray.rows, gray.cols, CV_32F);
    computeMagnitude(f, mag);
    
    // since magnitude value may be too greater to display, switch to logarithmic scale
    // M' = log(1+M)
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);
    
    // normalize
    cv::normalize(mag, mag, 0, 1, CV_MINMAX);
    
    cv::imshow("mag", mag);
    cv::waitKey(0);
    
    // shift to make low frequency in centre
    
    // crop spectrum to make sure row and col number both are even(use & 0xFE or & -2)
    mag = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2));
    
    // find center point
    const int cx = mag.cols / 2;
    const int cy = mag.rows / 2;
    
    // notice: we get reference here, not clone
    cv::Mat leftTop(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat leftBottom(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat rightTop(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat rightBottom(mag, cv::Rect(cx, cy, cx, cy));
    
    // re-organize
    cv::Mat tmp;
    leftTop.copyTo(tmp);
    rightBottom.copyTo(leftTop);
    tmp.copyTo(rightBottom);
    
    rightTop.copyTo(tmp);
    leftBottom.copyTo(rightTop);
    tmp.copyTo(leftBottom);
    
    cv::imshow("mag-reorganized", mag);
    cv::waitKey(0);
    
    //------------use opencv inside dft function
    
    cv::Mat gray1;
    cv::cvtColor(src, gray1, CV_BGR2GRAY);
    
    // 2n, 3n, 5n has the best compute efficency
    cv::Mat pad;
    int m = cv::getOptimalDFTSize(gray1.rows);
    int n = cv::getOptimalDFTSize(gray1.cols);
    cv::copyMakeBorder(gray1, pad, 0, m - gray1.rows, 0, n - gray1.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // pad for real part, zeros for imagine part
    cv::Mat planes[] = {cv::Mat_<float>(pad), cv::Mat::zeros(pad.size(), CV_32F)};
    cv::Mat complexVal;
    cv::merge(planes, 2, complexVal); // complexVal is two channel
    
    // compute dft
    cv::dft(complexVal, complexVal);
    
    // split to compute magnitude
    cv::split(complexVal, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat output = planes[0];
    
    output += cv::Scalar::all(1);
    cv::log(output, output);
    
    output = output(cv::Rect(0, 0, output.cols & -2, output.rows & -2));
    
    // find center point
    const int cx1 = output.cols / 2;
    const int cy1 = output.rows / 2;
    
    // notice: we get reference here, not clone
    cv::Mat leftTop1(output, cv::Rect(0, 0, cx1, cy1));
    cv::Mat leftBottom1(output, cv::Rect(0, cy1, cx1, cy1));
    cv::Mat rightTop1(output, cv::Rect(cx1, 0, cx1, cy1));
    cv::Mat rightBottom1(output, cv::Rect(cx1, cy1, cx1, cy1));
    
    // re-organize
    cv::Mat tmp1;
    leftTop1.copyTo(tmp1);
    rightBottom1.copyTo(leftTop1);
    tmp1.copyTo(rightBottom1);
    
    rightTop1.copyTo(tmp1);
    leftBottom1.copyTo(rightTop1);
    tmp1.copyTo(leftBottom1);
    
    cv::normalize(output, output, 0, 1, CV_MINMAX);
    
    cv::imshow("output-reorganized", output);
    cv::waitKey(0);
    
    return 0;
}
