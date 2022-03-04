import matplotlib.pyplot as plt
import pandas as pd   
import seaborn as sns
import natsort
import numpy as np
import csv
import os

def heat_map(Confusion_matrix, fig_save_dir):
    plt.rcParams['figure.figsize'] = [10, 8]
    sns.set(font_scale=3)
    sns.heatmap(Confusion_matrix,annot = True,  cmap = 'bone', fmt='d',linewidths=.5, vmin=0,vmax=200,xticklabels=['B1','B2','B3'], yticklabels=['B1','B2','B3'])
    plt.xlabel('Predicted Class', fontsize=30)
    plt.ylabel('Actual Class', fontsize=30)
    plt.savefig(fig_save_dir)
    plt.show()


def exe():
    directory = os.path.join(os.getcwd())
    ffile = os.listdir(directory)
    csv_list = [file for file in ffile if file.endswith("matrix.npy")]
    csv_list = natsort.natsorted(csv_list)
    
    for i in range(len(csv_list)):
        matrix=[]
        
        csv_name = csv_list[i]
        fig_name = '%s%s' % (csv_name,'.jpg')
        fig_save_dir = '%s%s%s' % (directory,'/',fig_name)
       #Confusion_matrix = [[1,2,3],[2,3,4],[3,4,5]]
        Confusion_matrix = np.load(csv_name)
        heat_map(Confusion_matrix, fig_save_dir)
