# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:38:59 2019

@author: Ismail
"""

import mnist_loader
import Digit_recognition

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Digit_recognition.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)