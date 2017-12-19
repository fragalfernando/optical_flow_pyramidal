#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 06:36:36 2017

@author: lfragago
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open("/home/lfragago/Documents/statistics.txt") as f:
    floats = map(float, f)

    
total = floats[0]
total = int(total)
floats = floats[1:]

maximum = np.zeros([total])
minimum = np.zeros([total])
mean = np.zeros([total])
median = np.zeros([total])


for i in range(total):
    minimum[i] = floats.pop(0)
    median[i] = floats.pop(0)
    maximum[i] = floats.pop(0)
    mean[i] = floats.pop(0)
    
fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0][0].plot(minimum)
ax[0][0].set_title('Minimum Error')
ax[0][0].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[0][0].set_ylabel('pixel error') 

ax[0][1].plot(median)
ax[0][1].set_title('Median Error')
ax[0][1].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[0][1].set_ylabel('pixel error') 

ax[1][0].plot(mean)
ax[1][0].set_title('Mean Error')
ax[1][0].set_xlabel('frame')
ax[1][0].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[1][0].set_ylabel('pixel error') 

ax[1][1].plot(maximum)
ax[1][1].set_title('Maximum Error')
ax[1][1].set_xlabel('frame')
ax[1][1].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[1][1].set_ylabel('pixel error') 

fig.savefig('graphs.jpg')
