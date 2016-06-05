#ifndef FUNCTIONS_HPP_INCLUDED
#define FUNCTIONS_HPP_INCLUDED
#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>

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

template<typename T>
void DoG_EdgeDetection(const cv::Mat& src, cv::Mat& dist, double tau, double sigmaE, double phi, int windowSize, int iteration) {
  // Perfom edge detection on lmninance channel
  assert(src.channels() == 1);
  dist = cv::Mat(src.rows, src.cols, CV_64FC1, 1.0);  

  // generate 2 Gaussian filtered image
  cv::Mat S_sigmaE, S_sigmaR, S_sigmaR2, temp;
  gaussianFilter2D<T>(src, windowSize, sigmaE*1.6, S_sigmaR2);

  // DoG
  for (int i = 0; i < iteration; ++i){
    gaussianFilter2D<T>(src, windowSize, sigmaE, S_sigmaE);
    gaussianFilter2D<T>(src, windowSize, sigmaE*sqrt(1.6), S_sigmaR);
    elementWiseOperator<T>(S_sigmaE, S_sigmaR, temp, [=](T x, T y){
      // slightly smoothed step function
      if ((x - tau * y) > 0)
        return 1.0;
      else
        return 1.0 + tanh(phi * (x - tau * y));
    });

    sigmaE *= sqrt(1.6);
    elementWiseOperator<T>(dist, temp, dist, [=](T x, T y){
      return std::min(x, y);
    });
    //cv::imshow("edge" + std::to_string(i), dist);
  }
  //cv::waitKey();
}

template<typename T>
void corre1D(const cv::Mat& src, const cv::Mat& kernel, cv::Mat& dist) {
  dist.create(src.rows, src.cols, src.type());
  cv::Mat padding;
  int paddingSize = kernel.cols / 2;
  paddingMirror<T>(src, paddingSize, padding);

  T num;
  T* ptrd; // pointer to dist
  const T* ptrk; // pointer to kernel
  const T* ptrp; // pointer to padding source
  for (int i = 0; i < dist.rows; i++) {
    ptrd = dist.ptr<T>(i);
    for (int j = 0; j < dist.cols; j++) {
      for (int c = 0; c < dist.channels(); c++) {
        num = 0.0;
        ptrk = kernel.ptr<T>(0);
        ptrp = padding.ptr<T>(i + paddingSize, j);
        for (int ki = 0; ki < kernel.cols; ki++) {
          num += (*ptrk++) * ptrp[c];
          ptrp += padding.channels();
        }
        *ptrd++ = num;
      }
    }
  }
}

template<typename T>
void sobelFilter3x3(const cv::Mat& src, cv::Mat& G, cv::Mat& Gx, cv::Mat& Gy) {
  G.create(src.rows, src.cols, src.type());
  Gx.create(src.rows, src.cols, src.type());
  Gy.create(src.rows, src.cols, src.type());
  cv::Mat sobel1 = (cv::Mat_<T>(1, 3) << 1, 2, 1);
  cv::Mat sobel2 = (cv::Mat_<T>(1, 3) << 1, 0, -1);
  
  // compute Gx
  corre1D<T>(src, sobel2, Gx);
  cv::transpose(Gx, Gx);
  corre1D<T>(Gx, sobel1, Gx);
  cv::transpose(Gx, Gx);

  // compute Gy
  corre1D<T>(src, sobel1, Gy);
  cv::transpose(Gy, Gy);
  corre1D<T>(Gy, sobel2, Gy);
  cv::transpose(Gy, Gy);

  // compute G = sqrt(Gx^2 + Gy^2)
  elementWiseOperator<T>(Gx, Gy, G, [](T x, T y) {return sqrt(x*x + y*y); });
}

template<typename T>
void imageBasedWarping(const cv::Mat& src, const cv::Mat& edgeMap, cv::Mat& dist, double sigmaS, double scale, int windowSize) {
  // Perform IBW on luminance channel only
  assert(src.channels() == 1);
  dist.create(edgeMap.rows, edgeMap.cols, edgeMap.type());
  
  cv::Mat G, Gx, Gy;
  sobelFilter3x3<T>(src, G, Gx, Gy);
  double minGx, maxGx;
  cv::minMaxLoc(Gx, &minGx, &maxGx);
  double minGy, maxGy;
  cv::minMaxLoc(Gy, &minGy, &maxGy);
  Gx = Gx / maxGx;
  Gy = Gy / maxGy;
  Gx = Gx * scale;
  Gy = Gy * scale;
  gaussianFilter2D<T>(Gx, windowSize, sigmaS, Gx);
  gaussianFilter2D<T>(Gy, windowSize, sigmaS, Gy);

  // inverse warping with bilinear interpolation
  T *ptrGx, *ptrGy, *ptrd;
  const T *ptre;
  T x0, x1, y0, y1;
  T posX, posY;
   for (int i = 0; i < dist.rows; i++) {
    ptrd = dist.ptr<T>(i);
    ptrGx = Gx.ptr<T>(i);
    ptrGy = Gy.ptr<T>(i);
    ptre = edgeMap.ptr<T>(i);
    for (int j = 0; j < dist.cols; j++) {
      if(ptrGx[0]==0.0 && ptrGy[0]==0.0)
        *ptrd++ = *ptre++;
      else if (j + ptrGx[0] < 0 || j + 1 + ptrGx[0] >= dist.cols)
        *ptrd++ = *ptre++;
      else if (i + ptrGy[0] < 0 || i + 1 + ptrGy[0] >= dist.rows)
        *ptrd++ = *ptre++;
      else {
        posX = j + floor(ptrGx[0]);
        posY = i + floor(ptrGy[0]);
        x1 = ptrGx[0] - floor(ptrGx[0]); x0 = ceil(ptrGx[0]) - ptrGx[0];
        y1 = ptrGy[0] - floor(ptrGy[0]); y0 = ceil(ptrGy[0]) - ptrGy[0];
        *ptrd++ = y0 * (x0 * (*edgeMap.ptr<T>(posY, posX)) + x1 * (*edgeMap.ptr<T>(posY, posX + 1))) 
                + y1 * (x0 * (*edgeMap.ptr<T>(posY + 1, posX)) + x1 * (*edgeMap.ptr<T>(posY + 1, posX + 1)));
        ptre++;
      }
      ptrGx++; ptrGy++;
    }
  }
}

// f(|x-y|) = exp(-|x-y|^2 / (2*sigma_s))
template<typename T>
void gaussianFilter2D(const cv::Mat& src, int windowSize, double sigmaS, cv::Mat& dist){
  cv::Mat kernel;
  getGaussianKernel<T>(1, windowSize, sigmaS, kernel); // 1D gaussian kernel

  cv::Mat temp(src.rows, src.cols, src.type());
  corre1D<T>(src, kernel, temp);
  cv::transpose(temp, temp);
  corre1D<T>(temp, kernel, temp);
  cv::transpose(temp, temp);
  dist = temp;
}

// bilateral filter
template<typename T>
void bilateralFilter(const cv::Mat& src, int windowSize, double sigmaS, double sigmaR, cv::Mat& dist){
  cv::Mat temp(src.rows, src.cols, src.depth(), 0.0);
  int paddingSize = windowSize / 2;
  cv::Mat gKernel;
  cv::Mat padding;
  getGaussianKernel<T>(windowSize, windowSize, sigmaS, gKernel);
  paddingWithReplicate<T>(src, paddingSize, padding);
  T* ptrd; // point to temp image
  const T* ptrgk; // point to gaussian kernel
  const T* ptrp; // point to padding image
  const T* ptrpr; // point to center point
  T sum, norm, mul;
  for (int i = 0; i < temp.rows; ++i){
    ptrd = temp.ptr<T>(i);
    for (int j = 0; j < temp.cols; ++j){
      for (int c = 0; c < temp.channels(); ++c){
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
  dist = temp;
}

// Fast bilateral filter, use segment to speed up
template<typename T>
void piecewiseLinearBilateralFilter(const cv::Mat& src, int windowSize, double sigmaS, double sigmaR, int segment, cv::Mat& dist){
  assert(src.channels() == 1);
  cv::Mat temp(src.rows, src.cols, src.depth(), 0.0);
  int paddingSize = windowSize / 2;

  cv::Mat intensity_exp_range;
  cv::Mat normalization_factor;
  cv::Mat H, J, H_gaussian;
  T currentIntensity = 0.0;
  T minIntensity = 0.0;
  T maxIntensity = 0.0;
  cv::minMaxIdx(src, &minIntensity, &maxIntensity);
  T step = (maxIntensity - minIntensity) / segment;
  for (int i = 0; i <= segment; i++){ // all intensity segment
    currentIntensity = minIntensity + i*step; // i
    elementWiseOperator<T>(src - currentIntensity, intensity_exp_range, [=](T x){return exp(-x*x / (2 * sigmaR*sigmaR)); }); // evaluate gr at each pixel, G = g_sigma_r(I - i)
    gaussianFilter2D<T>(intensity_exp_range, windowSize, sigmaS, normalization_factor); // normalization factor, K = G conv with gaussian kernel
    elementWiseOperator<T>(intensity_exp_range, src, H, [](T x, T y){return x*y; }); // compute H for each pixel, H = G x I (element-wise)
    gaussianFilter2D<T>(H, windowSize, sigmaS, H_gaussian); // H* = H conv with gaussian kernel
    elementWiseOperator<T>(H_gaussian, normalization_factor, J, [](T x, T y){return x / y; }); // normalize, H = H* / K (element-wise)
    elementWiseOperator<T>(J, src, J, [=](T x, T y){
      if (y > currentIntensity - step && y <= currentIntensity + step){
        return x * (step - abs(y - currentIntensity)) / step;
      }
      return 0.0;
    }); // InterpolationWeight
    temp = temp + J; // InterpolationWeight
  }
  dist = temp;
}

template<typename T>
void luminancePseudoQuantization(const cv::Mat& src, int bins, double bottom, double top, cv::Mat& dist){
  cv::Mat G, Gx, Gy;
  sobelFilter3x3<T>(src, G, Gx, Gy);

  // norm to [0,1]
  T minIntensity = 0.0;
  T maxIntensity = 0.0;
  cv::minMaxIdx(G, &minIntensity, &maxIntensity);
  elementWiseOperator<T>(G, G, [=](T x){return (x - minIntensity) / (maxIntensity - minIntensity); });

  // scale to [bottom,top]
  elementWiseOperator<T>(G, G, [=](T x){return x * (top - bottom); });
  elementWiseOperator<T>(G, G, [=](T x){return x + bottom; });

  // Pseudo Quantization
  cv::minMaxIdx(src, &minIntensity, &maxIntensity);
  // min, min+step, ..., max, total # of bins is bins
  T step = (maxIntensity - minIntensity) / (bins - 1);

  elementWiseOperator<T>(src, G, dist, [=](T x, T s){
    T q = round((x - minIntensity) / step)*step + minIntensity;
    return q + 0.5*step*tanh(s*(x - q));
  });
}

// Exact quantization, there are bins of value
template<typename T>
void luminanceQuantization(const cv::Mat& src, int bins, cv::Mat& dist){
  T minIntensity = 0.0;
  T maxIntensity = 0.0;
  cv::minMaxIdx(src, &minIntensity, &maxIntensity);

  // min, min+step, ..., max, total # of bins is bins
  T step = (maxIntensity - minIntensity) / (bins - 1);

  elementWiseOperator<T>(src, dist, [=](T x){
    return round((x - minIntensity) / step)*step + minIntensity;
  });
}

template<typename T>
void mergeImageAndEdge(const cv::Mat& src, const cv::Mat& edgeMap, double threshold, cv::Mat& dist){
  T minIntensity = 0.0;
  T maxIntensity = 0.0;
  cv::minMaxIdx(edgeMap, &minIntensity, &maxIntensity);
  elementWiseOperator<T>(src, edgeMap, dist, [=](T x, T y){
    if (y > threshold){
      return 0.0;
    }
    return x;
  });
}

}

#endif // FUNCTIONS_HPP_INCLUDED
