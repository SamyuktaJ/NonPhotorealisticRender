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
  cv::Mat differentColorSpace;

  // parameter
  struct {
    double windowSize;
    double sigmaS;
    double sigmaR;
  } bilateral;

  struct {
    double quantize;
    double edge;
  } iteration;

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
            bilateral.windowSize = std::stod(splstr[1]);
            bilateral.sigmaS = std::stod(splstr[2]);
            bilateral.sigmaR = std::stod(splstr[3]);
          }
          else if (splstr[0] == "iteration"){
            iteration.quantize = std::stod(splstr[1]);
            iteration.edge = std::stod(splstr[2]);
          }
        }
      }
    }
  }
  std::cout << "End parsing config file.\n" << std::endl;
}

void NonPhotorealisticRender::run(){
	cv::Mat dist;
	BGR2LAB(original, dist);
	cv::imshow("Lab", dist);
	LAB2BGR(dist, dist);
	cv::imshow("RGB", dist);
	cv::waitKey(0);
}

}

#endif // NON_PHOTOREALISTIC_RENDER_HPP_INCLUDED
