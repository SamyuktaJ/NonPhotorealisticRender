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
  for (int i = 0; i < input1.rows; ++i){
    const uchar *in1_ptr = input1.ptr(i);
    const uchar *in2_ptr = input2.ptr(i);
    for (int j = 0; j < input1.cols; ++j){
      for (int c = 0; c < input1.channels(); ++c){
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
  for (int i = 0; i < src.rows; ++i){
    const T* ptrs = src.ptr<T>(i);
    T* ptrd = temp.ptr<T>(i);
    for (int j = 0; j < src.cols; ++j){
      for (int c = 0; c < src.channels(); ++c){
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
  for (int i = 0; i < src1.rows; ++i){
    const T* ptrs1 = src1.ptr<T>(i);
    const T* ptrs2 = src2.ptr<T>(i);
    T* ptrd = temp.ptr<T>(i);
    for (int j = 0; j < src1.cols; ++j){
      for (int c = 0; c < src1.channels(); ++c){
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
  // convert to single-precision floating point (openCV constraint)
  cv::Mat src32 = cv::Mat(src.rows, src.cols, CV_32FC3);
  src.convertTo(src32, CV_32FC3);
  cv::Mat dist32 = cv::Mat(src.rows, src.cols, CV_32FC3);	
  cv::cvtColor(src32, dist32, CV_BGR2Lab);
  dist = cv::Mat(src.rows, src.cols, src.type());
  dist32.convertTo(dist, CV_64FC3);
}

// LAB->BGR
void LAB2BGR(const cv::Mat& src, cv::Mat& dist){
  // convert to single-precision floating point (openCV constraint)
  cv::Mat src32 = cv::Mat(src.rows, src.cols, CV_32FC3);
  src.convertTo(src32, CV_32FC3);
  cv::Mat dist32 = cv::Mat(src.rows, src.cols, CV_32FC3);
  cv::cvtColor(src32, dist32, CV_Lab2BGR);
  dist = cv::Mat(src.rows, src.cols, src.type());
  dist32.convertTo(dist, CV_64FC3);
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
template<typename T>
void paddingWithReplicate(const cv::Mat& src, int paddingSize, cv::Mat& dist){
  dist.create(src.rows + 2 * paddingSize, src.cols + 2 * paddingSize, src.type());
  const T* ptrs;
  T* ptrd;
  for (int i = 0; i < dist.rows; ++i){
    ptrd = dist.ptr<T>(i);
    // point to source Mat
    if (i - paddingSize < 0) { ptrs = src.ptr<T>(0); }
    else if (i - paddingSize >= src.rows) { ptrs = src.ptr<T>(src.rows - 1); }
    else { ptrs = src.ptr<T>(i - paddingSize); }

    for (int j = 0; j < dist.cols; ++j){
      for (int c = 0; c < dist.channels(); ++c){
        ptrd[c] = ptrs[c];
      }
      ptrd += dist.channels();
      if (j - paddingSize >= 0 && j - paddingSize < src.cols-1){
        ptrs += src.channels();
      }
    }
  }
}

// padding image with mirror
template<typename T>
void paddingMirror(const cv::Mat& src, int paddingSize, cv::Mat& dist) {
  dist = cv::Mat(src.rows + paddingSize * 2, src.cols + paddingSize * 2, src.type());
  T *destPix;
  for (int i = 0; i < dist.rows; ++i) {
    destPix = dist.ptr<T>(i);
    for (int j = 0; j < dist.cols; ++j) {
      for (int r = 0; r < src.channels(); ++r) {
        if (i < paddingSize && j < paddingSize)
          destPix[r] = src.ptr<T>(paddingSize - i - 1, paddingSize - j - 1)[r];
        else if (i < paddingSize && j >= (src.cols + paddingSize))
          destPix[r] = src.ptr<T>(paddingSize - i - 1, 2 * src.cols + paddingSize - j - 1)[r];
        else if (j < paddingSize && i >= (src.rows + paddingSize))
          destPix[r] = src.ptr<T>(2 * src.rows - i - 1 + paddingSize, paddingSize - j - 1)[r];
        else if (j >= (src.cols + paddingSize) && i >= (src.rows + paddingSize))
          destPix[r] = src.ptr<T>(2 * src.rows - i - 1 + paddingSize, 2 * src.cols + paddingSize - j - 1)[r];
        else if (j >= paddingSize && i < paddingSize)
          destPix[r] = src.ptr<T>(paddingSize - i - 1, j - paddingSize)[r];
        else if (j < paddingSize && i >= paddingSize)
          destPix[r] = src.ptr<T>(i - paddingSize, paddingSize - j - 1)[r];
        else if (j >= (src.cols + paddingSize) && i >= paddingSize)
          destPix[r] = src.ptr<T>(i - paddingSize, 2 * src.cols + paddingSize - j - 1)[r];
        else if (j >= paddingSize && i >= (src.rows + paddingSize))
          destPix[r] = src.ptr<T>(2 * src.rows - i - 1 + paddingSize, j - paddingSize)[r];
        else
          destPix[r] = src.ptr<T>(i - paddingSize, j - paddingSize)[r];
      }
      destPix += src.channels();
    }
  }
  
}

// f(|x-y|) = exp(-|x-y|^2 / (2*sigma_s))
template<typename T>
void getGaussianKernel(int height, int width, double sigmaS, cv::Mat& kernel){
  kernel.create(height, width, CV_64FC3);
  int heightRef = height / 2;
  int widthRef = width / 2;
  T* ptrk;
  T sum = 0.0;
  for (int i = 0; i < height; ++i){ // construct 2D gaussian kernel
    ptrk = kernel.ptr<T>(i);
    for (int j = 0; j < width; ++j){
      *ptrk = exp(-(pow(i - heightRef, 2) + pow(j - widthRef, 2)) / (2 * pow(sigmaS, 2)));
      sum += *ptrk;
      ptrk++;
    }
  }
  kernel = kernel / sum;
}

// f(|x-y|) = exp(-|x-y|^2 / (2*sigma_s))
template<typename T>
void GaussianFilter(const cv::Mat& src, cv::Mat& dist, int windowSize, double sigmaS){
  dist = cv::Mat(src.rows, src.cols, src.type());
  int step = windowSize / 2;
  cv::Mat srcPad;
  paddingMirror<T>(src, step, srcPad);

  // generate normalized 1-D Gaussian kernal
  std::vector<T> weight;
  for (int i = 0; i <= step; ++i)
    weight.push_back(exp(-0.5*(i*i / (sigmaS * sigmaS))));
  T weightSum = 0.0;
  for (int wy = -step; wy <= step; ++wy) {
    int wy_tmp = (wy >= 0) ? wy : -wy;
    weightSum += (weight[wy_tmp]);
  }
  for (int i = 0; i <= step; ++i)
    weight[i] = weight[i] / weightSum;

  // compute 2-D Gaussian filter with 1-D separatable Gaussian filter, O(n)
  cv::Mat GaussianTmp(srcPad.rows, srcPad.cols, srcPad.type());
  T *outPix;
  for (int i = 0; i < srcPad.rows; ++i) {
    outPix = GaussianTmp.ptr<T>(i);
    for (int j = 0; j < srcPad.cols; ++j) {
      for (int r = 0; r < srcPad.channels(); ++r) {
        outPix[0] = 0.0;
        if (j >= step && j < (srcPad.cols - step)) {
          for (int wx = -step; wx <= step; ++wx) {
            int wx_tmp = (wx >= 0) ? wx : -wx;
            outPix[0] += srcPad.ptr<T>(i, j + wx)[r] * weight[wx_tmp];
          }
        }
        outPix++;
      }
    }
  }

  for (int i = 0; i < dist.rows; ++i) {
    outPix = dist.ptr<T>(i);
    for (int j = 0; j < dist.cols; ++j) {
      for (int r = 0; r < srcPad.channels(); ++r) {
        outPix[0] = 0.0;
        for (int wy = -step; wy <= step; ++wy) {
          int wy_tmp = (wy >= 0) ? wy : -wy;
          outPix[0] += GaussianTmp.ptr<T>(i + step + wy, j + step)[r] * weight[wy_tmp];
        }
        outPix++;
      }
    }
  }
}

// bilateral filter
template<typename T>
void bilateralFilter(const cv::Mat& src, int windowSize, double sigmaS, double sigmaR, cv::Mat& dist){
  dist.create(src.rows, src.cols, src.type());
  int paddingSize = windowSize / 2;
  cv::Mat gKernel;
  cv::Mat padding;
  getGaussianKernel<T>(windowSize, windowSize, sigmaS, gKernel);
  paddingWithReplicate(src, paddingSize, padding);
  T* ptrd; // point to dist image
  const T* ptrgk; // point to gaussian kernel
  const T* ptrp; // point to padding image
  const T* ptrpr; // point to center point
  T sum, norm, mul;
  for (int i = 0; i < dist.rows; ++i){
    ptrd = dist.ptr<T>(i);
    for (int j = 0; j < dist.cols; ++j){
      for (int c = 0; c < dist.channels(); ++c){
        sum = 0.0;
        norm = 0.0;
        ptrpr = padding.ptr<T>(i + paddingSize, j + paddingSize);
        for (int ki = 0; ki < windowSize; ++ki){
          ptrp = padding.ptr<T>(i + ki, j);
          ptrgk = gKernel.ptr<T>(ki);
          for (int kj = 0; kj < windowSize; ++kj){
            mul = (*ptrgk) * exp(-(pow(ptrp[c] - ptrpr[c], 2)) / (2 * pow(sigmaR, 2))); // update kernel to bilateral
            sum += mul * ptrp[c];
            norm += mul;
            ptrgk++;
            ptrp += padding.channels();
          }
        }
        *ptrd++ = sum / norm; // normalize
      }
    }
  }
}

// Fast bilateral filter, use segment to speed up
template<typename T>
void piecewiseLinearBilateralFilter(const cv::Mat& src, int windowSize, double sigmaS, double sigmaR, int segment, cv::Mat& dist){
  // TODO
}

}

#endif // FUNCTIONS_HPP_INCLUDED
