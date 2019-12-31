# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:56:41 2019

@author: Ismail
"""

class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
a = [1,2,3,4,5]
print(a[:-1])
print(a[1:])