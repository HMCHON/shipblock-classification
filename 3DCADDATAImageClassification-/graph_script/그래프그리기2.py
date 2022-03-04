import pandas as pd                
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

mlp.rcParams['axes.unicode_minus'] = False
path = 'C:\\Users\lane\Anaconda3\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\batang.ttc'
fontprop = fm.FontProperties(fname=path, size=18)

plt.rcParams["font.family"] = 'batang'
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (14,4)

loc = 'C:\\Users\\lane\\Desktop\\학위논문\\graph\\Original_epoch'
path = os.listdir(loc)
accuracy = []

for i in range(len(path)):
    path2_name = '%s%s%s' % (loc,'\\',path[i])
    path2 = os.listdir(path2_name)
    for j in range(len(path2)):
        path3 = '%s%s%s' % (path2_name,'\\',path2[j])
        CSV = pd.read_csv(path3)
        CSV_accuracy = CSV['Accuracy']
        accuracy.append(CSV_accuracy[0])
        #CSV_val_acc = CSV['val_acc']
        #CSV_loss = CSV['loss']
        #CSV_acc = CSV['acc']


        #print(CSV)
DEN = accuracy[0:10]
RES = accuracy[10:20]
VGG = accuracy[20:30]



plt.figure(figsize=(30,20))

a = 5
b = 15
plt.plot(DEN,'k-o',linewidth = a,markersize = b, label = 'Densenet-201')
plt.plot(RES,'m-^', linewidth = a,markersize = b,label = 'Resnet-152V2')
plt.plot(VGG,'y-s', linewidth = a,markersize = b,label = 'VGG19')
plt.plot(1,0.68,markersize=30, c="b", lw=5, ls="--", marker="o", mec="k", mew=8, mfc="w")
plt.axis([-0.3,10,0,1])

for i in range(len(VGG)):
    number = str(VGG[i])
    plt.text(i,VGG[i]+0.01,s=number, fontsize=25)
for i in range(len(RES)):
    number = str(RES[i])
    plt.text(i,RES[i]+0.02,s=number, fontsize=25)
for i in range(len(DEN)):
    number = str(DEN[i])
    if i == 2:
       plt.text(i,DEN[i]+0.02,s=number, fontsize=25)
    else:
        plt.text(i,DEN[i]+0.02,s=number, fontsize=25)

plt.tick_params(axis='x', labelsize=30)
plt.xlabel('Epoch',fontsize=30)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10],
           [1,2,3,4,5,6,7,8,9,10])

plt.tick_params(axis='y', labelsize=30)
plt.ylabel('Prediction accuracy',fontsize=30)
plt.grid()
plt.legend(prop={'size': 30},loc='best')
plt.show()
#VGG = pd.read_csv('E:\새 폴더\graph\original\TL_DA_VGG19.csv')

