import pandas as pd                
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

import os

data = []
ac = []
ep = []
pred = []
train = []
pthr = ['25','30','35','40','45','50','55','60','65','70','75','80','85','90','95','100','105','110']
tthr = ['69','70','71','72','73','74','75','76','77','78','79','80','81','82','83']
epo=['1','2','3','4','5']

VGG = pd.read_csv('D:\\학위논문\\Result\\vgg-19_all_result.csv')
RES = pd.read_csv('D:\\학위논문\\Result\\resnet-152v2_all_result.csv')
DEN = pd.read_csv('D:\\학위논문\\Result\\densenet-201_all_result.csv')
ALL = pd.read_csv('D:\\학위논문\\Result\\69-83_resnet-152v2_all_result.csv')

target = ALL

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


ff= np.zeros((15,5,18,0)) #18개의 []가 한 묶음, 묶음 5개가 하나의 집합, 총 15개의 집합
ff = ff.tolist()

for i in range(len(f)):
    for j in range(len(tthr)):
        if (tthr[j] == f[i][1]):
            for e in range(len(epo)):
                if (epo[e] == f[i][3]):
                    for r in range(len(pthr)):
                        if (pthr[r] == f[i][0]):
                            ff[j][e][r].append(f[i])



plt.figure(figsize=(35,20))
plt.grid()

a = 1
b = 20
Graph = []
Graph1 = []
Graph2 = []
Graph3 = []

for i in range(15):
    for j in range(5):
        if j == 0: # 여기에서 원하는 Epoch 설정
            for r in range(18):
                graph1 = (ff[i][j][r][0][2])
                graph2 = (ff[i][j][r][0][0])
                graph3 = (ff[i][j][r][0][1])
                Graph1.append(graph1)
                Graph2.append(graph2)
                Graph3.append(graph3)

            if graph3 == '74':
                plt.plot(Graph2,Graph1,'k-o',linewidth = a,markersize = b, label = 'Bin 74 training image set')
            if graph3 == '75':
                plt.plot(Graph2,Graph1,'y-^',linewidth = a,markersize = b, label = 'Bin 75 training image set')
            if graph3 == '76':
                plt.plot(Graph2,Graph1,'k-*',linewidth = a,markersize = b, label = 'Bin 76 training image set')
            if graph3 == '77':
                plt.plot(Graph2,Graph1,'k-D',linewidth = a,markersize = b, label = 'Bin 77 training image set')
            if graph3 == '78':
                plt.plot(Graph2,Graph1,'k-s',linewidth = a,markersize = b, label = 'Bin 78 training image set')
            if graph3 == '79':
                plt.plot(Graph2,Graph1,'m-d',linewidth = a,markersize = b, label = 'Bin 79 training image set')
            if graph3 == '80':
                plt.plot(Graph2,Graph1,'m-1',linewidth = a,markersize = b, label = 'Bin 80 training image set')
            if graph3 == '81':
                plt.plot(Graph2,Graph1,'y-p',linewidth = a,markersize = b, label = 'Bin 81 training image set')
            if graph3 == '82':
                plt.plot(Graph2,Graph1,'m-X',linewidth = a,markersize = b, label = 'Bin 82 training image set')
            if graph3 == '83':
                plt.plot(Graph2,Graph1,'k-P',linewidth = a,markersize = b, label = 'Bin 83 training image set')

            plt.tick_params(axis='x', labelsize=30)
            plt.tick_params(axis='y', labelsize=30)
            plt.ylabel('Predicted accuracy', fontsize = 30)
            plt.xlabel('Bin i Prediction image set( 25 <= i <= 110, i = i+5 )',fontsize=30)
            
            plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                       ['i=25','i=30','i=35','i=40','i=45','i=50','i=55','i=60','i=65','i=70','i=75','i=80','i=85','i=90','i=95','i=100','i=105','i=110'])
            '''
            plt.xticks(['25','30','35','40','45','50','55','60','65','70','75','80','85','90','95','100','105','110'],
                       ['i=25','i=30','i=35','i=40','i=45','i=50','i=55','i=60','i=65','i=70','i=75','i=80','i=85','i=90','i=95','i=100','i=105','i=110'])
            '''
            mark1 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='o', markersize=30, label='Blue stars')
            mark2 = mlines.Line2D([], [], color='y',linestyle = 'None', marker='^', markersize=30, label='Blue stars')
            mark3 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='*', markersize=30, label='Blue stars')
            mark4 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='D', markersize=30, label='Blue stars')
            mark5 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='s', markersize=30, label='Blue stars')
            mark6 = mlines.Line2D([], [], color='m',linestyle = 'None', marker='d', markersize=30, label='Blue stars')
            mark7 = mlines.Line2D([], [], color='m',linestyle = 'None', marker='1', markersize=30, label='Blue stars')
            mark8 = mlines.Line2D([], [], color='y',linestyle = 'None', marker='p', markersize=30, label='Blue stars')
            mark9 = mlines.Line2D([], [], color='m',linestyle = 'None', marker='X', markersize=30, label='Blue stars')
            mark10 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='P', markersize=30, label='Blue stars')

            legend = plt.legend(handles = [mark1,mark2,mark3,mark4,mark5,mark6,mark7,mark8,mark9,mark10],
                                       labels = ('Bin 74 training image set','Bin 75 training image set','Bin 76 training image set','Bin 77 training image set','Bin 78 training image set','Bin 79 training image set','Bin 80 training image set','Bin 81 training image set','Bin 82 training image set','Bin 83 training image set'),
                                       prop={'size': 30}, bbox_to_anchor=(1, 0.25), loc='lower left')



            Graph1 = []
            Graph2 = []
            Graph3 = []

'''
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

'''













