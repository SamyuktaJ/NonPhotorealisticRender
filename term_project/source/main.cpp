#include "opencv2\opencv.hpp"

#include "NonPhotorealisticRender.hpp"

int main(int argc, char * argv[]){
  cp::NonPhotorealisticRender npr(argv[1]);
  npr.run();
  return 0;
}
