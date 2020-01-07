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

import pyqtgraph as pg
import numpy as np
from picoscope import ps4000
import sys

pg.setConfigOption('background', [247, 247, 247])
pg.setConfigOption('foreground', 'k')


def setupScope(nSamples=1 << 12):
    ps = ps4000.PS4000()

    # Example of simple capture
    res = ps.setSamplingFrequency(10E6, nSamples)
    sampleRate = res[0]
    print("Sampling @ %f MHz, %d samples" % (res[0] / 1E6, res[1]), f"RBW:{res[0]/(res[1]):.5f}")
    ps.setChannel("A", "DC",VRange=1e-3)
    return [ps, sampleRate]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

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
        #self.controlPanel.setLayout(self.controlPanelLayout)

        # Button for restarting average
        self.restartAvgBtn = QtWidgets.QPushButton("Restart average")
        self.restartAvgBtn.clicked.connect(self.restartAvg)

        # Label for showing the number of averaged traces
        self.avg_iter = 1
        self.avg_iter_label = QtWidgets.QLabel('Average of: ' + str(self.avg_iter))

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
        self.nSamples = 1 << 12
        self.nSamplesLabel = QtWidgets.QLabel('Number of samples')

        # Add window funtions
        self.wfComboBox = QtWidgets.QComboBox()
        self.wfComboBox.addItems(
            ['hamming', 'boxcar',  'blackman', 'bartlett', 'hanning', ])
        self.wfComboBoxLabel = QtWidgets.QLabel('Window Function')
        self.wfComboBox.currentIndexChanged[str].connect(self.changeWF)
        self.window = np.hamming(self.nSamples)

        # Stop button
        self.stopButton = QtWidgets.QPushButton('Stop averaging')
        self.stopButton.clicked.connect(self.stopAvg)
        self.avg = True

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
        self.saveButton = QtWidgets.QPushButton('Save trace')
        self.saveButton.clicked.connect(self.saveFile)

        # Add info label box
        self.infoLabel = QtWidgets.QLabel()
        self.infoLabel.setText(f'RBW: {10e6/self.nSamples:.1f} Hz')

        # Select Channel combobox
        self.channelComboBox = QtWidgets.QComboBox()
        self.channelLabel = QtWidgets.QLabel('Channel: ')
        self.channelComboBox.addItems(['A', 'B'])
        self.channelComboBox.currentIndexChanged.connect(self.changeChannel)
        self.channel = 0

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
        self.controlPanelLayout.addWidget(self.saveButton, 0, 4, 1, 2)

        self.controlPanelLayout.addWidget(self.infoLabel, 2, 4, 1, 2)


        # spacer = QtWidgets.QSpacerItem(
        #    1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        # self.controlPanelLayout.addItem(spacer, 20, 0)

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

        self.resize(1400, 800)
        self.show()

        # [self.scope, self.rate] = setupScope(self.nSamples)

        # self.scopeRunning = False

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

        self.prevSpec = 0
        self.lastValue = None
        self.lastData = None

        self.lastUpdate = None

        self.vb = self.specPlot.vb
        self._proxy = pg.SignalProxy(self.specPlot.scene(
        ).sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

        
    def mouseMoved(self, evt):
        pos = evt[0]  # using signal proxy turns original arguments into a tuple

        if self.specPlot.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            self.xLabel.setText(f'x: {mousePoint.x():.0f}')

    def changeChannel(self, s):
        self.channel = s

    def update(self):
        return
        # read data from device
        if self.scopeRunning is False:
            self.scope.runBlock()
            self.scopeRunning = True

        if self.scope.isReady():
            data = self.scope.getDataV(self.channel, self.nSamples)
            self.scopeRunning = False
        else:
            data = np.array([])
        if self.scopeRunning is False:
            self.scope.runBlock()

        if self.nSamples > 0 and data.shape == self.window.shape and self.avg:
            # update spectrum
            spec = np.abs(np.fft.fft(data*self.window)[:self.nSamples//2])
            # spec[:self]
            # https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft
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

                x = np.linspace(0, self.rate / 2., len(spec))
                
                self.specPlot.plot(x, spec**2/self.rate/self.nSamples,
                                   pen=pg.mkPen(color=(0, 0, 0, 180)), clear=True)

                for key in self.maxHoldList.keys():
                    self.specPlot.plot(np.linspace(0, self.rate / 2., len(self.maxHoldList[key])),
                                   self.maxHoldList[key]**2/self.rate/self.nSamples, pen=pg.mkPen(color=self.colorPens[key]))

                self.prevSpec = spec
                self.lastData = data
                self.avg_iter_label.setText(
                    'Average of: ' + str(self.avg_iter))
            except ValueError:
                self.prevSpec = np.zeros_like(spec)

    def setAvgType(self, s):
        self.avgType = s
        self.restartAvg()

    def restartAvg(self):
        self.prevSpec = 0
        self.avg_iter = 0
        self.avg = True

    def stopAvg(self):
        self.avg = False

    def setNumAvg(self, s):
        self.avgAlpha = 2/(s+1)

    def setNSamples(self, s):
        self.nSamples = self.nSamplesList[s]
        res = self.scope.setSamplingFrequency(10E6, self.nSamples)
        self.rate = res[0]
        self.resetMaxHold()
        self.changeWF(self.wfComboBox.currentText().__str__())
        self.infoLabel.setText(f'RBW: {self.rate /self.nSamples:.1f} Hz')

    def changeWF(self, s):
        if s == 'boxcar':
            self.window = np.ones(self.nSamples)
        elif s == 'hamming':
            self.window = np.hamming(self.nSamples)
        elif s == 'blackman':
            self.window = np.blackman(self.nSamples)
        elif s == 'bartlett':
            self.window = np.bartlett(self.nSamples)
        elif s == 'hanning':
            self.window = np.hanning(self.nSamples)
        self.restartAvg()

    def setMaxHold(self, color):
        self.maxHoldList[color] = self.prevSpec


    def resetMaxHold(self):
        self.maxHoldList = {'r': np.array(
            []), 'g': np.array([]), 'b': np.array([])}

    def saveFile(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "QFileDialog.getSaveFileName()", "", "All Files (*);;Text Files (*.txt)", options=options)
        if fileName == '':
            return
        x = np.linspace(0, self.rate / 2., len(self.prevSpec))
        outputArray = np.append(x[:, None], self.prevSpec[:, None]**2/self.rate/self.nSamples , axis=1)
        np.savetxt(fileName, outputArray)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = MainWindow()
    sys.exit(app.exec_())