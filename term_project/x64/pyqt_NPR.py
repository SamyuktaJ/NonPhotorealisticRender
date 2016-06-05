# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from PyQt4 import QtCore, QtGui
import subprocess
from subprocess import DEVNULL
import glob

class ParameterSlider(QtGui.QWidget):
    def __init__(self, name, minValue, maxValue, interval, initialValue, parent = None):
        super(ParameterSlider, self).__init__(parent)
        self.paraName = QtGui.QLabel()
        self.paraName.setText(name)
        self.paraList = np.arange(minValue, maxValue+interval, interval)

        self.value = QtGui.QLabel()

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, len(self.paraList)-1)
        temp = abs(self.paraList - initialValue)
        self.slider.setValue(np.where(temp<0.0001)[0][0])

        self.layout = QtGui.QHBoxLayout()
        self.layout.addWidget(self.paraName)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.value)

        self.slider.valueChanged.connect(self.valuechange)
        self.valuechange()

    def valuechange(self):
        idx = self.slider.value()
        self.value.setText(str(self.paraList[idx]))

    def getValue(self):
        return self.paraList[self.slider.value()]

def write_config_file(**para):
    with open(para['filename'], 'w') as file:
        file.write('originalImage,{}{}\n'.format(para['image_dir'], para['image']))
        file.write('bilateral,{windowSize},{sigmaS},{sigmaR},{segment},{skip}\n'.format(**para['bilateral']))
        file.write('iteration,{quantize},{edge}\n'.format(**para['iteration']))
        file.write('quantization,{bins},{bottom},{top}\n'.format(**para['quantization']))
        file.write('DoG,{windowSize},{sigmaE},{tau},{phi},{iteration}\n'.format(**para['DoG']))
        file.write('IBW,{windowSize},{sigmaS},{scale}\n'.format(**para['IBW']))

def run_NPR(**para):
    write_config_file(**para)
    cmd = '{} {}'.format(para['exe'], para['filename'])
    subprocess.call(cmd)

class Dictionary(QtGui.QWidget):
    def __init__(self, parent = None):
        super(Dictionary, self).__init__(parent)
        self.setWindowTitle("Non Photorealistic Render")
        self.resize(1280, 720)
        self.createLayout()

        self.getfile()
        
    def createLayout(self):
        self.bilateral = {
            'name': QtGui.QLabel('Bilateral'),
            'windowSize': ParameterSlider('Window Size', 1, 101, 2, 31),
            'sigmaS': ParameterSlider('Sigma S', 0.1, 10, 0.1, 3),
            'sigmaR': ParameterSlider('Sigma R', 0.1, 10, 0.1, 3),
            'segment': ParameterSlider('Segment', 1, 50, 1, 21)
        }
        self.bilateral['name'].setAlignment(QtCore.Qt.AlignCenter)

        self.iteration = {
            'name': QtGui.QLabel('Iteration'),
            'quantize': ParameterSlider('Quantize', 1, 10, 1, 3),
            'edge': ParameterSlider('Edge', 1, 10, 1, 3)
        }
        self.iteration['name'].setAlignment(QtCore.Qt.AlignCenter)

        self.quantization = {
            'name': QtGui.QLabel('Quantization'),
            'bins': ParameterSlider('Bins', 1, 20, 1, 7),
            'bottom': ParameterSlider('Bottom', 0.1, 10, 0.1, 0.7),
            'top': ParameterSlider('Top', 0.1, 10, 0.1, 1.4),
        }
        self.quantization['name'].setAlignment(QtCore.Qt.AlignCenter)

        self.dog = {
            'name': QtGui.QLabel('DoG'),
            'windowSize': ParameterSlider('Window Size', 1, 101, 2, 7),
            'sigmaE': ParameterSlider('Sigma E', 0.1, 10, 0.1, 0.5),
            'tau': ParameterSlider('Tau', 0.8, 1.2, 0.01, 0.98),
            'phi': ParameterSlider('Phi', 0.1, 10, 0.1, 1.0),
            'iteration': ParameterSlider('Iteration', 1, 10, 1, 3)
        }
        self.dog['name'].setAlignment(QtCore.Qt.AlignCenter)

        self.ibw = {
            'name': QtGui.QLabel('IBW'),
            'windowSize': ParameterSlider('Window Size', 1, 101, 2, 7),
            'sigmaS': ParameterSlider('Sigma S', 0.1, 10, 0.1, 1.5),
            'scale': ParameterSlider('Scale', 0.1, 5, 0.1, 1.5)
        }
        self.ibw['name'].setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.bilateral['name'])
        layout.addLayout(self.bilateral['windowSize'].layout)
        layout.addLayout(self.bilateral['sigmaS'].layout)
        layout.addLayout(self.bilateral['sigmaR'].layout)
        layout.addLayout(self.bilateral['segment'].layout)

        layout.addWidget(self.iteration['name'])
        layout.addLayout(self.iteration['quantize'].layout)
        layout.addLayout(self.iteration['edge'].layout)

        layout.addWidget(self.quantization['name'])
        layout.addLayout(self.quantization['bins'].layout)
        layout.addLayout(self.quantization['bottom'].layout)
        layout.addLayout(self.quantization['top'].layout)

        layout.addWidget(self.dog['name'])
        layout.addLayout(self.dog['windowSize'].layout)
        layout.addLayout(self.dog['sigmaE'].layout)
        layout.addLayout(self.dog['tau'].layout)
        layout.addLayout(self.dog['phi'].layout)
        layout.addLayout(self.dog['iteration'].layout)

        layout.addWidget(self.ibw['name'])
        layout.addLayout(self.ibw['windowSize'].layout)
        layout.addLayout(self.ibw['sigmaS'].layout)
        layout.addLayout(self.ibw['scale'].layout)

        ## checkbox
        ckb = QtGui.QHBoxLayout()
        self.b1 = QtGui.QCheckBox("IBW")
        self.b1.setChecked(True)
        self.b1.stateChanged.connect(self.showNPRImage)
        ckb.addWidget(self.b1)

        self.b2 = QtGui.QCheckBox("Quantize")
        self.b2.setChecked(True)
        self.b2.stateChanged.connect(self.showNPRImage)
        ckb.addWidget(self.b2)
        
        self.skip = QtGui.QCheckBox("Skip bilateral")
        self.skip.setChecked(False)
        ckb.addWidget(self.skip)

        layout.addLayout(ckb)

        ## botton
        self.quit = QtGui.QPushButton('&Quit')
        self.quit.clicked.connect(self.quitNPR)

        self.run = QtGui.QPushButton('&Run')
        self.run.clicked.connect(self.runNPR)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(self.quit)
        btns.addWidget(self.run)

        layout.addLayout(btns)

        ## image
        pic = QtGui.QVBoxLayout()
        self.btn = QtGui.QPushButton('&Open Image File')
        self.btn.clicked.connect(self.getfile)
        self.le = QtGui.QLabel()
        self.lea = QtGui.QLabel()
        pic.addWidget(self.btn)
        pic.addWidget(self.le)
        pic.addWidget(self.lea)

        warp = QtGui.QHBoxLayout()
        warp.addLayout(layout)
        warp.addLayout(pic)

        self.setLayout(warp)

    def getfile(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', os.getcwd(), 'Image files (*.jpg *.png)')
        self.filename = fname
        self.le.setPixmap(QtGui.QPixmap(fname).scaledToHeight(512))
        self.moveResult()
    
    def quitNPR(self):
        self.moveResult()
        self.close()      
        
    def moveResult(self):
        name = os.path.splitext(os.path.basename(self.filename))[0]
        dirname = 'Result_' + name
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        cmd = 'mv {}*.png {}'.format(name, dirname)
        subprocess.call(cmd)
        
    def runNPR(self):
        self.moveResult()
        if self.skip.isChecked() == True:
            skip = 1
        else:
            skip = 0
        para = {
            'filename': 'conf.csv',
            'exe': 'term_project.exe',
            'image_dir': '',
            'image': self.filename,
            ## bilateral, {windowSize}, {sigmaS}, {sigmaR}, {segment}
            'bilateral': {
                'windowSize': self.bilateral['windowSize'].getValue(),
                'sigmaS': float('%.6g'%(self.bilateral['sigmaS'].getValue())),
                'sigmaR': float('%.6g'%(self.bilateral['sigmaR'].getValue())),
                'segment': self.bilateral['segment'].getValue(),
                'skip': skip},
            ## iteration, {quantize}, {edge}
            'iteration': {
                'quantize': self.iteration['quantize'].getValue(),
                'edge': self.iteration['edge'].getValue()},
            ## quantization, {bins}, {bottom}, {top}
            'quantization': {
                'bins': self.quantization['bins'].getValue(),
                'bottom': float('%.6g'%(self.quantization['bottom'].getValue())),
                'top': float('%.6g'%(self.quantization['top'].getValue()))},
            ## edge detection (DoG),{windowSize}, {sigmaE}, {tau}, {phi}, {iteration}
            'DoG': {
                'windowSize': self.dog['windowSize'].getValue(),
                'sigmaE': float('%.6g'%(self.dog['sigmaE'].getValue())),
                'tau': float('%.6g'%(self.dog['tau'].getValue())),
                'phi': float('%.6g'%(self.dog['phi'].getValue())),
                'iteration': self.dog['iteration'].getValue()},
            ## image based warping (IBW), {windowSize}, {sigmaS}, {scale}
            'IBW': {
                'windowSize': self.ibw['windowSize'].getValue(),
                'sigmaS': float('%.6g'%(self.ibw['sigmaS'].getValue())),
                'scale': float('%.6g'%(self.ibw['scale'].getValue()))}
        }
        run_NPR(**para)
        self.showNPRImage()

    def showNPRImage(self):
        name = os.path.splitext(os.path.basename(self.filename))[0]
        pat = '{}*'.format(name)
        if self.b2.isChecked() == True:
            pat += '_quantize'
        else:
            pat += '_filtered'
        if self.b1.isChecked() == True:
            pat += '_ibw'
        else:
            pat += '_edge'
        pat += '.png'
        fname = glob.glob(pat)
        self.lea.setPixmap(QtGui.QPixmap(fname[0]).scaledToHeight(512))

app = QtGui.QApplication(sys.argv)

dictionary = Dictionary()
dictionary.show()

app.exec_()
