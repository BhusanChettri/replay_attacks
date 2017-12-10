import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_entropy_loss(train_loss, val_loss, savePath):
        
    x_axis = np.arange(1, len(train_loss)+1)
    plt.xticks(x_axis, x_axis, rotation=50)
    
    plt.plot(x_axis, train_loss, 'o--', label='train_ce')
    plt.plot(x_axis, val_loss, 'o--',label='validation_ce')
    plt.xlabel('# Training Epochs')
    plt.ylabel('Avg CE loss')
    plt.legend()        
    plt.savefig(savePath) 
    plt.close()

    
    
def plot_2dGraph(xlabel, ylabel, train_loss, val_loss, xlegend, ylegend, savePath):
    x_axis = np.arange(1, len(train_loss)+1)
    plt.xticks(x_axis, x_axis, rotation=50)
    
    plt.plot(x_axis, train_loss, 'o--', label=xlegend)
    plt.plot(x_axis, val_loss, 'o--',label=ylegend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()        
    plt.savefig(savePath)    
    plt.close()

def plot1d_data(data, title, savePath):    
    plt.plot(data, '.',) 
    plt.title(title)
    plt.savefig(savePath)
    plt.close()
    
    
def plot_1dGraph(xlabel, ylabel, loss, xlegend, savePath):
        
    x_axis = np.arange(1, len(loss)+1)
    plt.xticks(x_axis, x_axis, rotation=50)
    
    plt.plot(x_axis, loss, 'o--', label=xlegend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()        
    plt.savefig(savePath)
    plt.close()
    
def plot_BatchData(data, savePath):
    
    x_axis = np.arange(1, len(data)+1)
    plt.xticks(x_axis, x_axis, rotation=50)
    
    plt.plot(data, '.')
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    #plt.legend()
    plt.title('Data distribution in a batch (300x1025=30750 values)')
    plt.savefig(savePath)
    plt.close()
    
def plotStats(maxDataList, minDataList, avgDataList, savePath, title):
        
    x_axis = np.arange(1, len(maxDataList)+1)
    plt.xticks(x_axis, x_axis, rotation=50)
    
    plt.plot(x_axis, maxDataList, 'o--', label='max')
    plt.plot(x_axis, minDataList, 'o--',label='min')
    plt.plot(x_axis, avgDataList, 'o--',label='avg')
    
    plt.legend()        
    #plt.title('Stats plot of data distribution in a batch (300x1025=30750 values)')
    plt.title(title)
    plt.savefig(savePath)
    plt.close()
    
def plot_label_Stats(genLabels, spoofLabels, savePath, title):
        
    x_axis = np.arange(1, len(genLabels)+1)
    plt.xticks(x_axis, x_axis, rotation=50)
    
    plt.plot(x_axis, genLabels, 'o--', label='genuine')
    plt.plot(x_axis, spoofLabels,'o--',label='spoof')
    plt.xlabel('#batches')
    plt.ylabel('#genuine and spoofed')
        
    plt.legend()        
    plt.title(title)
    plt.savefig(savePath)    
    plt.close()