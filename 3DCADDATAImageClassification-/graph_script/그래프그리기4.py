import pandas as pd                
import matplotlib as mlp
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams["font.family"] = 'batang'
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (14,4)



data = []
ac = []
ep = []
pred = []
train = []
pthr = ['25','30','35','40','45','50','55','60','65','70','75','80','85','90','95','100','105','110']
tthr = ['69','70','71','72','73','74','75','76','77','78','79','80','81','82','83']

VGG = pd.read_csv('D:\\학위논문\\Result\\69-83_vgg-19_all_result.csv')
RES = pd.read_csv('D:\\학위논문\\Result\\69-83_resnet-152v2_all_result.csv')
DEN = pd.read_csv('D:\\학위논문\\Result\\69-83_densenet-201_all_result.csv')

target = VGG

for i in range(len(target)):
    CSV = target.loc[i]
    case_name = CSV['case']
    acc = CSV['Accuracy']

    if (len(case_name)==36):
        epoch = case_name[28:29]
    else:
        epoch = case_name[29:30]
    
    ac.append(acc)
    ep.append(epoch)
    data.append((case_name,acc,epoch))

for i in range(len(target)):
    p = data[i][0]

    if (len(p)==36):
        p = p[9:11]
    else:
        p = p[9:12]
    
    pred.append(p)

for i in range(len(target)):
    t = data[i][0]
    if (len(t)==36):
        t = t[34:36]
    else:
        t = t[35:37]
    
    train.append(t)


f = []    
for i in range(len(target)):
    final = (pred[i],train[i],ac[i],ep[i])
    f.append(final)

import sys

ff= np.zeros((15,18,0)) #18개의 []가 한 묶음, 묶음 5개가 하나의 집합, 총 15개의 집합
ff = ff.tolist()
maxmax=f[0]

for i in range(len(f)):
    for j in range(len(tthr)):
        if (tthr[j] == f[i][1]):
            for r in range(len(pthr)):
                if (pthr[r] == f[i][0]):
                    ff[j][r].append(f[i])
                    #print((tthr[j],f[i][1],pthr[r],f[i][0]))

plt.figure(figsize=(30,20))
plt.grid()


a = 1
b = 3
Graph1 = []
Graph2 = []
Graph11 = [0,0,0,0,0]
Graph22 = [0,0,0,0,0]
check = []

for i in range(15):
    for j in range(18):
        for r in range(5):
            graph1 = (int(ff[i][j][r][3]))
            Graph1.append(graph1)
            graph2 = (ff[i][j][r][2])
            Graph2.append(graph2)
        for r in range(len(Graph1)):
            #print(Graph[r])
            if Graph1[r]==1:
                Graph11[0]=Graph1[r]
                Graph22[0]=Graph2[r]
            if Graph1[r]==2:
                Graph11[1]=Graph1[r]
                Graph22[1]=Graph2[r]
            if Graph1[r]==3:
                Graph11[2]=Graph1[r]
                Graph22[2]=Graph2[r]
            if Graph1[r]==4:
                Graph11[3]=Graph1[r]
                Graph22[3]=Graph2[r]
            if Graph1[r]==5:
                Graph11[4]=Graph1[r]
                Graph22[4]=Graph2[r]
                
        plt.plot(Graph11,Graph22,'k-o',linewidth = a,markersize = b, label = 'VGG19 validation loss')
        plt.xticks([1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
                   ['1','','2','','3','','4','','5'])
        plt.tick_params(axis='x', labelsize=30)
        plt.xlabel('Epoch',fontsize=30)
        plt.ylabel('Predicted Accuracy ', fontsize = 30)
        plt.tick_params(axis='y', labelsize=30)
        check.append((Graph11,Graph22))
        Graph1 = []
        Graph2 = []
        
plt.plot(int(maxmax[3]),float(maxmax[2]),markersize=15,c="k", lw=5, ls="--", marker="o", mec="k", mew=5, mfc="w")





















