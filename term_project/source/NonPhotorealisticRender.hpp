#ifndef NON_PHOTOREALISTIC_RENDER_HPP_INCLUDED
#define NON_PHOTOREALISTIC_RENDER_HPP_INCLUDED
#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "functions.hpp"

namespace cp{

class NonPhotorealisticRender{
public:
  explicit NonPhotorealisticRender(const std::string& configFile);
  void run();
private:
  std::string imageFilename;
  cv::Mat original;

  // parameter
  struct {
    int windowSize;
    double sigmaS;
    double sigmaR;
    int segment;
  } bilateral;

  struct {
    int quantize;
    int edge;
  } iteration;

  struct {
    int bins;
    double bottom;
    double top;
  } quantization;

  struct {
    int windowSize;
    double tau;
    double sigmaE;
    double phi;
  } DoG;

  struct {
    double sigmaS;
    double scale;
    int windowSize;
  } IBW;

  // help function
  const std::vector<std::string> Token(const std::string& in, const char delimeter){
    std::istringstream ss(in);
    std::vector<std::string> elements;
    std::string item;
    while (getline(ss, item, delimeter)){
      elements.push_back(item);
    }
    return elements;
  }
};

NonPhotorealisticRender::NonPhotorealisticRender(const std::string& configFile){
  std::fstream fs;
  fs.open(configFile.c_str(), std::ios::in);
  if (!fs.is_open()){
    std::cerr << "Error: config file doesn't exist." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  else{
    std::cout << "Parsing config file \"" << configFile << "\"" << std::endl;
    std::string raw_line;
    while (getline(fs, raw_line)){
      std::vector<std::string> splstr = Token(raw_line, ',');
      if (splstr.size() > 0){
        if (raw_line[0] == '#'){
          //std::cout << raw_line << std::endl;
        }
        else{
          if (splstr[0] == "originalImage"){
            std::cout << "Reading original image " << splstr[1] << std::endl;
            readImage(splstr[1], original);
            if (original.empty()){
              std::cerr << "Error: image file doesn't exist. " << splstr[1] << std::endl;
              std::exit(EXIT_FAILURE);
            }

            // get filename
            splstr = Token(splstr[1], '/');
            splstr = Token(splstr.back(), '.');
            imageFilename = splstr.front();
          }
          else if (splstr[0] == "bilateral"){
            bilateral.windowSize = std::stoi(splstr[1]);
            bilateral.sigmaS = std::stod(splstr[2]);
            bilateral.sigmaR = std::stod(splstr[3]);
            bilateral.segment = std::stoi(splstr[4]);
          }
          else if (splstr[0] == "iteration"){
            iteration.quantize = std::stoi(splstr[1]);
            iteration.edge = std::stoi(splstr[2]);
          }
          else if (splstr[0] == "quantization"){
            quantization.bins = std::stoi(splstr[1]);
            quantization.bottom = std::stod(splstr[2]);
            quantization.top = std::stod(splstr[3]);
          }
          else if (splstr[0] == "DoG") {
            DoG.windowSize = std::stoi(splstr[1]);
            DoG.sigmaE = std::stod(splstr[2]);
            DoG.tau = std::stod(splstr[3]);
            DoG.phi = std::stod(splstr[4]);
          }
          else if (splstr[0] == "IBW") {
            IBW.windowSize = std::stoi(splstr[1]);
            IBW.sigmaS = std::stod(splstr[2]);
            IBW.scale = std::stod(splstr[3]);
          }
        }
      }
    }
  }
  std::cout << "End parsing config file.\n" << std::endl;
}

void NonPhotorealisticRender::run(){
  // change color space
  cv::Mat differentColorSpace;
  BGR2LAB(original, differentColorSpace);

  // separate luminance
  std::vector<cv::Mat> mv;
  cv::Mat luminance, luminanceFiltered;
  cv::split(differentColorSpace, mv);
  mv[0].copyTo(luminance);
  mv[0].copyTo(luminanceFiltered);

  // recursion bilateral filter
  int times = std::max(iteration.quantize, iteration.edge);
  cv::Mat forQuan, forEdge;
  for (int i = 0; i < times; ++i){
    piecewiseLinearBilateralFilter<double>(luminanceFiltered, bilateral.windowSize, bilateral.sigmaS, bilateral.sigmaR, bilateral.segment, luminanceFiltered);

    if (i == iteration.quantize - 1){
      luminanceFiltered.copyTo(forQuan);
    }
    if (i == iteration.edge - 1){
      luminanceFiltered.copyTo(forEdge);
    }
  }

  // luminance quantization
  cv::Mat quantize;
  //luminanceQuantization<double>(forQuan, quantization.bins, quantize);
  luminancePseudoQuantization<double>(forQuan, quantization.bins, quantization.bottom, quantization.top, quantize);

  // edge detection
  cv::Mat edge, edgeIBW;
  DoG_EdgeDetection<double>(forEdge, edge, DoG.tau, DoG.sigmaE, DoG.phi, DoG.windowSize);

  // image based warping
  imageBasedWarping<double>(luminance, edge, edgeIBW, IBW.sigmaS, IBW.scale, IBW.windowSize);

  cv::imshow("luminance", luminance / 100.0);
  cv::imshow("srcFiltered", forEdge / 100.0);
  cv::imshow("edge", edge);
  cv::imshow("edgeIBW", edgeIBW);
  cv::waitKey(0);
}

}

#endif // NON_PHOTOREALISTIC_RENDER_HPP_INCLUDED
