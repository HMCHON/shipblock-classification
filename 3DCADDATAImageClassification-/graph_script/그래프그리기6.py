import pandas as pd                
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os

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
epo=['1','2','3','4','5']

VGG = pd.read_csv('D:\\학위논문\\Result\\69-83_vgg-19_all_result.csv')
RES = pd.read_csv('D:\\학위논문\\Result\\69-83_resnet-152v2_all_result.csv')
DEN = pd.read_csv('D:\\학위논문\\Result\\69-83_densenet-201_all_result.csv')

target = RES

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

A = []
X = []
Y = []

for i in range(15):
    for r in range(5):
        for j in range(18):
            a = ff[i][r][j][0][2] # 정확도
            A.append(a)
    A = np.array(A)
    value = np.sum(A)
    value2 = value/(18*5)
    X.append(i)
    Y.append(value2)
    print(value2)
    A = []






plt.figure(figsize=(30,20))
plt.grid()

a = 1
b = 3
Graph = []
Graph1 = []
Graph2 = []
Graph3 = []


for i in range(18):
    for j in range(5):
        if j >= 0:
            for r in range(15):
                graph1 = (ff[r][j][i][0][2])
                graph3 = (ff[r][j][i][0][0])
                graph2 = (ff[r][j][i][0][1])
                Graph1.append(graph1)
                Graph2.append(graph2)
                Graph3.append(graph3)
                    
            
            plt.plot(Graph2,Graph1,'k-o',linewidth = a,markersize = b, label = 'Bin 74 training image set')
            plt.plot(Y,'k-o',linewidth=10, markersize=b)
            plt.plot(Y,'w--o',linewidth=4, markersize=b)

            plt.tick_params(axis='x', labelsize=30)
            plt.tick_params(axis='y', labelsize=30)
            plt.ylabel('Predicted accuracy', fontsize = 30)
            plt.xlabel('Bin i training image set( 69 <= i <= 83, i = i+1 )',fontsize=30)
            plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                       ['i=69','i=70','i=71','i=72','i=73','i=74','i=75','i=76','i=77','i=78','i=79','i=80','i=81','i=82','i=83'])
            
            
            Graph1 = []
            Graph2 = []
            Graph3 = []







