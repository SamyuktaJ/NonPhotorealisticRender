#ifndef FUNCTIONS_HPP_INCLUDED
#define FUNCTIONS_HPP_INCLUDED
#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include <math.h>

namespace cp{

//Multi channel 8bit PSNR
double PSNR_UCHAR(const cv::Mat& input1, const cv::Mat& input2){
  double diff;
  double sigma = 0;
  for (int i = 0; i < input1.rows; i++){
    const uchar *in1_ptr = input1.ptr(i);
    const uchar *in2_ptr = input2.ptr(i);
    for (int j = 0; j < input1.cols; j++){
      for (int c = 0; c < input1.channels(); c++){
        diff = (*in1_ptr++) - (*in2_ptr++);
        sigma += diff*diff;
      }
    }
  }
  double mse = sigma / input1.rows / input1.cols;
  if (mse < 1e-100) return 10000;
  else return 20 * log10(255) - 10 * log10(mse);
}

// dist = func(src)
template<typename T>
void elementWiseOperator(const cv::Mat& src, cv::Mat& dist, const std::function<T(T)>& func){
  cv::Mat temp(src.rows, src.cols, src.type());
  for (int i = 0; i < src.rows; i++){
    const T* ptrs = src.ptr<T>(i);
    T* ptrd = temp.ptr<T>(i);
    for (int j = 0; j < src.cols; j++){
      for (int c = 0; c < src.channels(); c++){
        *ptrd++ = func(*ptrs++);
      }
    }
  }
  dist = temp;
}

// dist = func(src1, src2)
template<typename T>
void elementWiseOperator(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& dist, const std::function<T(T, T)>& func){
  assert(src1.type() == src2.type());
  cv::Mat temp(src1.rows, src1.cols, src1.type());
  for (int i = 0; i < src1.rows; i++){
    const T* ptrs1 = src1.ptr<T>(i);
    const T* ptrs2 = src2.ptr<T>(i);
    T* ptrd = temp.ptr<T>(i);
    for (int j = 0; j < src1.cols; j++){
      for (int c = 0; c < src1.channels(); c++){
        *ptrd++ = func(*ptrs1++, *ptrs2++);
      }
    }
  }
  dist = temp;
}

// read image, normalize to [0, 1], use CV_64FC3
void readImage(const std::string& filename, cv::Mat& image){
  image = cv::imread(filename);
  image.convertTo(image, CV_64FC3, 1.0 / 255);
}

// write image, scale to [0, 255], flag to control gamma correction
void writeImage(const std::string& filename, const cv::Mat& image){
  cv::Mat output;
  image.convertTo(output, CV_8UC3, 255);
  cv::imwrite(filename, output);
}

// change color space, note that opencv use BGR
// BGR->LAB
void BGR2LAB(const cv::Mat& src, cv::Mat& dist){
  // TODO
}

// LAB->BGR
void LAB2BGR(const cv::Mat& src, cv::Mat& dist){
  // TODO
}

// BGR->YUV
void BGR2YUV(const cv::Mat& src, cv::Mat& dist){
  // TODO
}

// YUV->BGR
void YUV2BGR(const cv::Mat& src, cv::Mat& dist){
  // TODO
}

// padding image with boundary
void paddingWithReplicate(const cv::Mat& src, int paddingSize, cv::Mat& dist){
  // TODO
}

void paddingMirror(const cv::Mat& src, int paddingSize, cv::Mat& dist) {
	dist = cv::Mat(src.rows + paddingSize * 2, src.cols + paddingSize * 2, src.type());
	double *destPix;
	for (int i = 0; i < dist.rows; i++) {
		destPix = dist.ptr<double>(i);
		for (int j = 0; j < dist.cols; j++) {
			for (int r = 0; r < src.channels(); r++) {
				if (i < paddingSize && j < paddingSize)
					destPix[r] = src.ptr<double>(paddingSize - i - 1, paddingSize - j - 1)[r];
				else if (i < paddingSize && j >= (src.cols + paddingSize))
					destPix[r] = src.ptr<double>(paddingSize - i - 1, 2 * src.cols + paddingSize - j - 1)[r];
				else if (j < paddingSize && i >= (src.rows + paddingSize))
					destPix[r] = src.ptr<double>(2 * src.rows - i - 1 + paddingSize, paddingSize - j - 1)[r];
				else if (j >= (src.cols + paddingSize) && i >= (src.rows + paddingSize))
					destPix[r] = src.ptr<double>(2 * src.rows - i - 1 + paddingSize, 2 * src.cols + paddingSize - j - 1)[r];
				else if (j >= paddingSize && i < paddingSize)
					destPix[r] = src.ptr<double>(paddingSize - i - 1, j - paddingSize)[r];
				else if (j < paddingSize && i >= paddingSize)
					destPix[r] = src.ptr<double>(i - paddingSize, paddingSize - j - 1)[r];
				else if (j >= (src.cols + paddingSize) && i >= paddingSize)
					destPix[r] = src.ptr<double>(i - paddingSize, 2 * src.cols + paddingSize - j - 1)[r];
				else if (j >= paddingSize && i >= (src.rows + paddingSize))
					destPix[r] = src.ptr<double>(2 * src.rows - i - 1 + paddingSize, j - paddingSize)[r];
				else
					destPix[r] = src.ptr<double>(i - paddingSize, j - paddingSize)[r];
			}
			destPix += src.channels();
		}
	}
	
}

// f(|x-y|) = exp(-|x-y|^2 / (2*sigma_s))
void GaussianFilter(const cv::Mat& src, cv::Mat& dist, int windowSize, double sigmaS){
	int step = windowSize / 2;
	std::vector<double> weight;
	for (int i = 0; i <= step; i++)
		weight.push_back(exp(-0.5*(i*i / (sigmaS * sigmaS))));
	double weightSum = 0.0;
	for (int wy = -step; wy <= step; wy++) {
		int wy_tmp = (wy >= 0) ? wy : -wy;
		weightSum += (weight[wy_tmp]);
	}
	for (int i = 0; i <= step; i++)
		weight[i] = weight[i] / weightSum;
}

// bilateral filter
void bilateralFilter(const cv::Mat& src, int windowSize, double sigmaS, double sigmaR, cv::Mat& dist){
  // TODO
}

// Fast bilateral filter, use segment to speed up
void piecewiseLinearBilateralFilter(const cv::Mat& src, int windowSize, double sigmaS, double sigmaR, int segment, cv::Mat& dist){
  // TODO
}

}

#endif // FUNCTIONS_HPP_INCLUDED
