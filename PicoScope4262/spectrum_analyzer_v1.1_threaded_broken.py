#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:52:18 2019

@author: Jonas Mathiassen
A PyQt spectrum analyzer gui for the PicoScope 4262. 
Requires the picoscope library found at https://github.com/colinoflynn/pico-python.
This in turn requires the Picoscope SDK found at https://www.picotech.com/downloads.
"""
from PyQt5 import QtWidgets, QtCore
from picoscope import ps4000
import pyqtgraph as pg

import numpy as np
import sys
import os
import datetime

if os.name == 'nt':  # This should correspond to windows
    pyrep = r'Y:\\membrane\\Programs\\Python'
    data_path = r'Z:\membrane\test-setup\data\\'
    prefix = 'Y:\\'
else:
    if sys.platform == 'linux':
        pyrep = '/quarpi/other/membrane/Programs/Python'  # for Linux
        data_path = '/quarpi/data/membrane/test-setup/data/'
        prefix = '/'
    else:
        pyrep = '/Volumes/other/membrane/Programs/Python'  # for Mac
        prefix = '/'
if not(pyrep in sys.path):
    sys.path.append(pyrep)

(y, m, d) = str(datetime.date.today()).split('-')

path = os.path.join(data_path, y,m,d)

if not os.path.exists(path):
    os.makedirs(path)

pg.setConfigOption('background', [247, 247, 247])
pg.setConfigOption('foreground', 'k')


def setupScope(nSamples=1 << 12, voltageRange=2, channel=0):
    ps = ps4000.PS4000()

    # Example of simple capture
    res = ps.setSamplingFrequency(10E6, nSamples)
    sampleRate = res[0]
    print("Sampling @ %f MHz, %d samples" % (res[0] / 1E6, res[1]),
          f"RBW:{res[0]/(res[1]):.5f}")
    ps.setChannel(channel, "DC",VRange=voltageRange)
    return [ps, sampleRate]


class acquisitionWorker(QtCore.QThread):
    signal = QtCore.pyqtSignal('PyQt_PyObject')
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        

        # Initialize parameters and then scope
        self.scopeRunning = False
        self.prevSpec = 0
        self.lastData = None
        self.avg = True
        self.avgType = "Average"
        self.lastUpdate = None
        self.nSamples = 1 << 12
        self.voltageRange = 2
        self.window = np.hanning(self.nSamples)
        self.channel = 0
        self.voltageClipCounter = 0
        self.avg_iter = 1
        self.avg_iter_tmp = 1
        [self.scope, self.rate] = setupScope(self.nSamples,
                                     self.voltageRange,
                                     self.channel)

    def work(self):
        while True:
            
            if self.avg_iter_tmp == None:
                pass
            else:
                self.avg_iter = self.avg_iter_tmp
                self.avg_iter_tmp = None
                
            # read data from device
            if self.scopeRunning is False:
                self.scope.runBlock()
                self.scopeRunning = True
                self.scope.waitReady()
            try:
                if self.scope.isReady():
                    data = self.scope.getDataV(self.channel, self.nSamples)
                    self.scopeRunning = False
                else:
                    data = np.array([])
                if self.scopeRunning is False:
                    self.scope.runBlock()
            except OSError:
                print('OS error, scope hanging')
                continue
            if self.nSamples > 0 and data.shape == self.window.shape and self.avg:
                # update spectrum
                spec = np.abs(np.fft.fft(data*self.window)[:self.nSamples//2])
                
                if max(data) > self.voltageRange:
                    self.voltageClipCounter += 1
                    self.statusBar.showMessage(
                            f'Voltage Clipped {self.voltageClipCounter} times')
                # Take the average
                try:
                    if self.avgType == "Average":
                        spec = (self.avg_iter*self.prevSpec + spec) / \
                            (self.avg_iter + 1)
                        self.avg_iter += 1
                    elif self.avgType == "Exponential Rolling Average" and self.prevSpec is not 0:
                        spec = self.avgAlpha*spec + \
                            (1-self.avgAlpha) * self.prevSpec
                        if self.avg_iter < self.numAvgSpinBox.value():
                            self.avg_iter += 1
                    
                    self.signal.emit(spec)
                except ValueError:
                    self.prevSpec = np.zeros_like(spec)
            

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        
        
        self.worker = acquisitionWorker()
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.worker.signal.connect(self.updatePlot)
        
        self.thread.start()
        self.init_gui()
        
        self.vb = self.specPlot.vb
        self._proxy = pg.SignalProxy(self.specPlot.scene().sigMouseMoved,
                                     rateLimit=60, slot=self.mouseMoved)

        
    def mouseMoved(self, evt):
        pos = evt[0]  # using signal proxy turns original arguments into a tuple

        if self.specPlot.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            self.xLabel.setText(f'x: {mousePoint.x():.0f}\t\t\t y: {10**mousePoint.y():.2e}')

    def changeChannel(self, s):
        self.worker.channel = s
        
    def changeVoltageRange(self, s):
        self.worker.voltageRange = self.vRangeList[s]
        self.worker.scope.stop()
        self.worker.scope.setChannel(self.worker.channel, "DC",
                                     VRange=self.worker.voltageRange)
        self.statusBar.showMessage('')
        self.voltageClipCounter = 0
        
    def updatePlot(self, data):
        
        spec = data
        x = np.linspace(0, self.rate / 2., len(spec))
        zOrder = 5
        curve = self.specPlot.plot(x, spec**2/self.rate/self.nSamples,
                           pen=pg.mkPen(color=(0, 0, 0, 180)), clear=True)
        curve.setZValue(zOrder)
        for key in self.maxHoldList.keys():
            zOrder -= 1
            curve = self.specPlot.plot(np.linspace(0, self.worker.rate / 2.,
                                                   len(self.maxHoldList[key])),
                           self.maxHoldList[key]**2/self.worker.rate/ \
                           self.worker.nSamples,
                           pen=pg.mkPen(color=self.colorPens[key]))
            curve.setZValue(zOrder)
        self.avg_iter_label.setText(
            'Average of: ' + str(self.worker.avg_iter))


    def setAvgType(self, s):
        self.worker.avgType = s
        self.restartAvg()

    def restartAvg(self):
        self.worker.prevSpec = 0
        self.worker.avg_iter_tmp = 0
        self.worker.avg = True

    def stopAvg(self):
        self.worker.avg = False

    def setNumAvg(self, s):
        self.worker.avgAlpha = 2/(s+1)

    def setNSamples(self, s):
        self.worker.nSamples = self.nSamplesList[s]
        res = self.worker.scope.setSamplingFrequency(10E6, self.worker.nSamples)
        self.worker.rate = res[0]
#        self.resetMaxHold() Hopefully this will not break the code
        self.changeWF(self.wfComboBox.currentText().__str__())
        self.infoLabel.setText(f'RBW: {self.worker.rate /self.worker.nSamples:.1f} Hz')

    def changeWF(self, s):
        # Changes the window function
        if s == 'boxcar':
            self.worker.window = np.ones(self.worker.nSamples)
        elif s == 'hamming':
            self.worker.window = np.hamming(self.worker.nSamples)
        elif s == 'blackman':
            self.worker.window = np.blackman(self.worker.nSamples)
        elif s == 'bartlett':
            self.worker.window = np.bartlett(self.worker.nSamples)
        elif s == 'hanning':
            self.worker.window = np.hanning(self.worker.nSamples)
        self.restartAvg()

    def setMaxHold(self, color):
        self.maxHoldList[color] = self.worker.prevSpec


    def resetMaxHold(self):
        self.maxHoldList = {'r': np.array(
            []), 'g': np.array([]), 'b': np.array([])}

    def saveFile(self):
        fileName = self.fileNameTextBox.text() 
        if fileName == '':
            return
        x = np.linspace(0, self.worker.rate / 2., len(self.prevSpec))
        outputArray = np.append(x[:, None], self.worker.prevSpec[:, None]**2/\
                                self.worker.rate/self.worker.nSamples , axis=1)
        outputDict = {'data': outputArray,
                      'window function': self.worker.window,
                      'voltage range': self.worker.voltageRange,
                      'number of samples': self.worker.avg_iter,
                      }
        np.savetxt(path+os.path.sep+fileName+'.csv', outputArray)
        
    def init_gui(self):
        
        # This is the top level widget
        self.centralWidget = QtWidgets.QWidget()

        # Make grid layout and add button + graph into it
        self.layout = QtWidgets.QGridLayout()

        # Add a grid inside the QGridLayout
        self.centralWidget.setLayout(self.layout)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('Picoscope Spectrum Analyzer')

        # Make a control panel widget that holds all buttons and such.
        #self.controlPanel = QtWidgets.QWidget()
        self.controlPanelLayout = QtWidgets.QGridLayout()


        # Button for restarting average
        self.restartAvgBtn = QtWidgets.QPushButton("Restart average")
        self.restartAvgBtn.clicked.connect(self.restartAvg)

        # Label for showing the number of averaged traces
        self.avg_iter_label = QtWidgets.QLabel('Average of: ' + str(self.worker.avg_iter))

        # Combobox for choosing how to average
        self.avgComboBox = QtWidgets.QComboBox()
        self.avgComboBox.addItems(
            ["Average", "Exponential Rolling Average", "Continuous"])
        self.avgComboBox.currentIndexChanged[str].connect(self.setAvgType)
        self.avgType = "Average"
        self.numAvgSpinBox = QtWidgets.QSpinBox()
        self.numAvgSpinBox.setRange(1, 10000)
        self.numAvgSpinBox.setValue(10)
        self.avgAlpha = 2/(self.numAvgSpinBox.value()+1)
        # Choose number of points to do in rolling average
        self.numAvgSpinBoxLabel = QtWidgets.QLabel(
            'Number of exponential averages')
        self.numAvgSpinBox.valueChanged.connect(self.setNumAvg)

        # Choose number of samples
        self.nSamplesList = [1 << i for i in range(8, 21)]
        self.nSamplesListLabels = [f'{i:.3e}' for i in self.nSamplesList]
        self.nSamplesComboBox = QtWidgets.QComboBox()
        self.nSamplesComboBox.addItems(self.nSamplesListLabels)
        self.nSamplesComboBox.currentIndexChanged.connect(self.setNSamples)

        self.nSamplesLabel = QtWidgets.QLabel('Number of samples')

        # Add window funtions
        self.wfComboBox = QtWidgets.QComboBox()
        self.wfComboBox.addItems(
            ['hanning', 'boxcar',  'blackman', 'bartlett', 'hamming', ])
        self.wfComboBoxLabel = QtWidgets.QLabel('Window Function')
        self.wfComboBox.currentIndexChanged[str].connect(self.changeWF)


        # Stop button
        self.stopButton = QtWidgets.QPushButton('Stop averaging')
        self.stopButton.clicked.connect(self.stopAvg)

        # Max hold buttons
        self.maxHoldWidget = QtWidgets.QWidget()
        self.maxHoldLayout = QtWidgets.QHBoxLayout()
        # self.maxHoldWidget.setLayout(self.maxHoldLayout)

        self.maxHoldButtonGreen = QtWidgets.QPushButton('Max Hold')
        self.maxHoldButtonGreen.setStyleSheet("background-color: green")
        self.maxHoldButtonGreen.clicked.connect(lambda: self.setMaxHold('g'))
        
        self.maxHoldButtonBlue = QtWidgets.QPushButton('Max Hold')
        self.maxHoldButtonBlue.setStyleSheet("background-color: blue")
        self.maxHoldButtonBlue.clicked.connect(lambda: self.setMaxHold('b'))

        self.maxHoldButtonRed = QtWidgets.QPushButton('Max Hold')
        self.maxHoldButtonRed.setStyleSheet("background-color: red")
        self.maxHoldButtonRed.clicked.connect(lambda: self.setMaxHold('r'))

        self.resetMaxHoldButton = QtWidgets.QPushButton('Reset max hold')
        self.resetMaxHoldButton.clicked.connect(self.resetMaxHold)

        self.maxHoldLayout.addWidget(self.maxHoldButtonGreen,)
        self.maxHoldLayout.addWidget(self.maxHoldButtonBlue)
        self.maxHoldLayout.addWidget(self.maxHoldButtonRed)

        self.maxHoldList = {'r': np.array(
            []), 'g': np.array([]), 'b': np.array([])}
        self.colorPens = {'r': (200, 0, 0, 180), 'g': (0, 200, 0, 180), 'b': (0, 0, 200, 180)}
        # Add save button
        self.fileNameLabel = QtWidgets.QLabel('File name')
        self.fileNameTextBox = QtWidgets.QLineEdit()
        self.saveButton = QtWidgets.QPushButton('Save trace')
        self.saveButton.clicked.connect(self.saveFile)

        # Add info label box
        self.infoLabel = QtWidgets.QLabel()
        self.infoLabel.setText(f'RBW: {10e6/self.worker.nSamples:.1f} Hz')

        # Select Channel combobox
        self.channelComboBox = QtWidgets.QComboBox()
        self.channelLabel = QtWidgets.QLabel('Channel: ')
        self.channelComboBox.addItems(['A', 'B'])
        self.channelComboBox.currentIndexChanged.connect(self.changeChannel)


        # Select voltage range
        self.voltageRangeComboBox = QtWidgets.QComboBox()
        self.voltageRangeLabel = QtWidgets.QLabel('Voltage range: ')
        self.vRangeList = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1,1, 2, 5, 10, 20]
        self.vRangeListStr = [str(x)+'V' for x in self.vRangeList]
        self.voltageRangeComboBox.addItems(self.vRangeListStr)
        self.voltageRangeComboBox.currentIndexChanged.connect(
                self.changeVoltageRange)
        
        # Add status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Add widgets to layout
        self.controlPanelLayout.addWidget(self.channelComboBox, 0,1, 1,1)
        self.controlPanelLayout.addWidget(self.channelLabel, 0, 0, 1, 1)
        # Add combobox for choosng type of average
        self.controlPanelLayout.addWidget(self.avgComboBox, 1, 0, 1, 1)
        # Add Stop-button
        self.controlPanelLayout.addWidget(self.stopButton, 1, 1, 1, 1)
        # Add button to restart average
        self.controlPanelLayout.addWidget(self.restartAvgBtn, 2, 1,)
        self.controlPanelLayout.addWidget(self.avg_iter_label, 2, 0)
        # Add spinbox for choosing number of exponential averages
        self.controlPanelLayout.addWidget(self.numAvgSpinBoxLabel, 3, 0)
        self.controlPanelLayout.addWidget(self.numAvgSpinBox, 3, 1)

        # Add sample size function
        self.controlPanelLayout.addWidget(self.nSamplesLabel, 0, 2)
        self.controlPanelLayout.addWidget(self.nSamplesComboBox, 0, 3)
        # Add window function stuff to control panel
        self.controlPanelLayout.addWidget(self.wfComboBoxLabel, 1, 2)
        self.controlPanelLayout.addWidget(self.wfComboBox, 1, 3)
        # Add max-hold buttons
        self.controlPanelLayout.addLayout(self.maxHoldLayout, 2, 2, 1, 2)

        self.controlPanelLayout.addWidget(self.resetMaxHoldButton, 3, 2, 1, 2)
        # Save Button
        self.controlPanelLayout.addWidget(self.fileNameLabel, 0, 4, 1, 2)
        self.controlPanelLayout.addWidget(self.fileNameTextBox, 1, 4, 1, 2)
        self.controlPanelLayout.addWidget(self.saveButton, 2, 4, 1, 1)
        self.controlPanelLayout.addWidget(self.infoLabel, 3, 4, 1, 2)
        
        # Select voltage range button
        self.controlPanelLayout.addWidget(self.voltageRangeLabel, 0, 6,1,1)
        self.controlPanelLayout.addWidget(self.voltageRangeComboBox, 1, 6, 1, 1)
        
        # Add control panel to grid.
        self.layout.addLayout(self.controlPanelLayout, 4, 0, 1, 1)

        # Add plot to grid
        self.plotWidget = pg.GraphicsLayoutWidget()

        self.xLabel = pg.LabelItem(justify='left')

        self.layout.addWidget(self.plotWidget, 0, 0, 4, 1)
        self.plotWidget.addItem(self.xLabel, row=1)
        
        self.specPlot = self.plotWidget.addPlot(
            row=0, col=0, labels={'bottom': ('Frequency', 'Hz'), 'left': '<font>V<sup>2</sup>/Hz<\font>'})

        self.specPlot.setLogMode(y=True)
        self.specPlot.showGrid(x=True, y=True, alpha=0.15)
        
        self.voltageRangeComboBox.setCurrentIndex(7)

        self.resize(1400, 800)
        self.show()
    

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = MainWindow()
#    sys.exit(app.exec_())