## How to use with GUI

Type below line in command line
> python -u pyqt_NPR.py

Or double click run.bat

## How to use with command line

> term_project.exe {config_file} or pull config file into HW3_Matting.exe

#### EX:
> term_project.exe config.csv

### How to set config file
Separate with ','

A line start with '#' is a comment

#### EX:

> \# Non Photorealistic Render Configuration

> originalImage,../image/Lenna.png

> \# bilateral, {windowSize}, {sigmaS}, {sigmaR}, {segment}, {skip}

> bilateral,31,3,4.25,21,0

> \# iteration, {quantize}, {edge}

> iteration,3,3

> \# quantization, {bins}, {bottom}, {top}

> quantization,7,0.9,1.8

> \# edge detection (DoG),{windowSize}, {sigmaE}, {tau}, {phi}, {iteration}

> DoG,21,0.5,0.98,3.0,5

> \# image based warping, {windowSize}, {sigmaS}, {scale}

> IBW,21,0.8,2.7
