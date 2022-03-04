import pandas as pd                
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import natsort
import os

plt.rcParams["font.family"] = 'batang'
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (14,4)



vgg_loc = 'D:\\학위논문\\graph\\bin\\vgg'
den_loc = 'D:\\학위논문\\graph\\bin\\den'
res_loc = 'D:\\학위논문\\graph\\bin\\res'

path = [vgg_loc, den_loc, res_loc]
loss = []
acc = []
val_loss = []
val_acc = []

for i in range(3):
    pathpath = path[i]
    pathpathpath = os.listdir(pathpath)
    pathpathpath = natsort.natsorted(pathpathpath, reverse=False)
    for j in range(15):
        path3 = '%s%s%s' % (pathpath,'\\',pathpathpath[j])
        CSV = pd.read_csv(path3)
    
        CSV_loss = CSV['loss']
        CSV_acc = CSV['acc']
        CSV_val_loss = CSV['val_loss']
        CSV_val_acc = CSV['val_acc']
        
        loss.append(CSV_loss)
        acc.append(CSV_acc)
        val_loss.append(CSV_val_loss)
        val_acc.append(CSV_val_acc)
    if i == 0:
        vgg_loss = loss
        vgg_acc = acc
        vgg_val_loss = val_loss
        vgg_val_acc = val_acc
    if i == 1:
        den_loss = loss
        den_acc = acc
        den_val_loss = val_loss
        den_val_acc = val_acc
    if i == 2:
        res_loss = loss
        res_acc = acc
        res_val_loss = val_loss
        res_val_acc = val_acc
    
    loss = []
    acc = []
    val_loss = []
    val_acc = []

CSVa = den_acc
CSVl = den_loss
CSVva = den_val_acc
CSVvl = den_val_loss

#plt.plot(1,0.68,markersize=30, c="b", lw=5, ls="--", marker="o", mec="k", mew=8, mfc="w")

plt.figure(figsize=(30,20))
plt.grid()
a = 3
b = 20
c =3

j = 69
i=0
for i in range(15):
    if i == 0:
        color = 'k-^'
    if i == 1:
        color = 'y-o'
    if i == 2:
        color = 'c-s'
    if i == 3:
        color = 'm-d'
    if i == 4:
        color = 'g-*'
    if i == 5:
        color = 'k-p'
    if i == 6:
        color = 'm-h'
    if i == 7:
        color = 'c-X'     
        
    if i == 8:
        color = 'k-'
        marker = "^"
        mec="k"
        mew=c
        mfc = "w"
    if i == 9:
        color = 'y-'
        marker = "o"
        mec="y"
        mew=c
        mfc = "w"
    if i == 10:
        color = 'c-'
        marker = "s"
        mec="c"
        mew=c
        mfc = "w"
    if i == 11:
        color = 'm-'
        marker = "d"
        mec="m"
        mew=c
        mfc = "w"
    if i == 12:
        color = 'g-'
        marker = "*"
        mec="g"
        mew=c
        mfc = "w"
    if i == 13:
        color = 'k-'
        marker = 'p'
        mec = 'k'
        mew=c
        mfc = 'w'
    if i== 14:
        color = 'm-'
        marker = 'h'
        mec = 'm'
        mew=c
        mfc = 'w'

    j = j+1
    if i <= 7:
        plt.plot(CSVa[i],color,linewidth = a,markersize = b)
    else:
        plt.plot(CSVa[i],color,linewidth = a,markersize = b, marker=marker, mec=mec, mew=mew, mfc=mfc)
    
j = 69
i=0
for i in range(15):
    if i == 0:
        color = 'k--^'
    if i == 1:
        color = 'y--o'
    if i == 2:
        color = 'c--s'
    if i == 3:
        color = 'm--d'
    if i == 4:
        color = 'g--*'
    if i == 5:
        color = 'k--p'
    if i == 6:
        color = 'm--h'
    if i == 7:
        color = 'c--X'     
        
    if i == 8:
        color = 'k--'
        marker = "^"
        mec="k"
        mew=c
        mfc = "w"
    if i == 9:
        color = 'y--'
        marker = "o"
        mec="y"
        mew=c
        mfc = "w"
    if i == 10:
        color = 'c--'
        marker = "s"
        mec="c"
        mew=c
        mfc = "w"
    if i == 11:
        color = 'm--'
        marker = "d"
        mec="m"
        mew=c
        mfc = "w"
    if i == 12:
        color = 'g--'
        marker = "*"
        mec="g"
        mew=c
        mfc = "w"
    if i == 13:
        color = 'k--'
        marker = 'p'
        mec = 'k'
        mew=c
        mfc = 'w'
    if i== 14:
        color = 'm--'
        marker = 'h'
        mec = 'm'
        mew=c
        mfc = 'w'
    j = j+1
    if i <= 7:
        plt.plot(CSVl[i],color,linewidth = a,markersize = b)
    else:
        plt.plot(CSVl[i],color,linewidth = a,markersize = b, marker=marker, mec=mec, mew=mew, mfc=mfc)
    
j = 69
i=0
for i in range(15):
    if i == 0:
        color = 'k:^'
    if i == 1:
        color = 'y:o'
    if i == 2:
        color = 'c:s'
    if i == 3:
        color = 'm:d'
    if i == 4:
        color = 'g:*'
    if i == 5:
        color = 'k:p'
    if i == 6:
        color = 'm:h'
    if i == 7:
        color = 'c:X'     
        
    if i == 8:
        color = 'k:'
        marker = "^"
        mec="k"
        mew=c
        mfc = "w"
    if i == 9:
        color = 'y:'
        marker = "o"
        mec="y"
        mew=c
        mfc = "w"
    if i == 10:
        color = 'c:'
        marker = "s"
        mec="c"
        mew=c
        mfc = "w"
    if i == 11:
        color = 'm:'
        marker = "d"
        mec="m"
        mew=c
        mfc = "w"
    if i == 12:
        color = 'g:'
        marker = "*"
        mec="g"
        mew=c
        mfc = "w"
    if i == 13:
        color = 'k:'
        marker = 'p'
        mec = 'k'
        mew=c
        mfc = 'w'
    if i== 14:
        color = 'm:'
        marker = 'h'
        mec = 'm'
        mew=c
        mfc = 'w'
    j = j+1
    if i <= 7:
        plt.plot(CSVva[i],color,linewidth = a,markersize = b)
    else:
        plt.plot(CSVva[i],color,linewidth = a,markersize = b, marker=marker, mec=mec, mew=mew, mfc=mfc)

j=69
i=0
for i in range(15):
    if i == 0:
        color = 'k-.^'
    if i == 1:
        color = 'y-.o'
    if i == 2:
        color = 'c-.s'
    if i == 3:
        color = 'm-.d'
    if i == 4:
        color = 'g-.*'
    if i == 5:
        color = 'k-.p'
    if i == 6:
        color = 'm-.h'
    if i == 7:
        color = 'c-.X'     
        
    if i == 8:
        color = 'k-.'
        marker = "^"
        mec="k"
        mew=c
        mfc = "w"
    if i == 9:
        color = 'y-.'
        marker = "o"
        mec="y"
        mew=c
        mfc = "w"
    if i == 10:
        color = 'c-.'
        marker = "s"
        mec="c"
        mew=c
        mfc = "w"
    if i == 11:
        color = 'm-.'
        marker = "d"
        mec="m"
        mew=c
        mfc = "w"
    if i == 12:
        color = 'g-.'
        marker = "*"
        mec="g"
        mew=c
        mfc = "w"
    if i == 13:
        color = 'k-.'
        marker = 'p'
        mec = 'k'
        mew=c
        mfc = 'w'
    if i== 14:
        color = 'm-.'
        marker = 'h'
        mec = 'm'
        mew=c
        mfc = 'w'
    j = j+1
    if i <= 7:
        plt.plot(CSVvl[i],color,linewidth = a,markersize = b)
    else:
        plt.plot(CSVvl[i],color,linewidth = a,markersize = b, marker=marker, mec=mec, mew=mew, mfc=mfc)


mark1 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='^', markersize=30, label='Blue stars')
mark2 = mlines.Line2D([], [], color='y',linestyle = 'None', marker='o', markersize=30, label='Blue stars')
mark3 = mlines.Line2D([], [], color='c',linestyle = 'None', marker='s', markersize=30, label='Blue stars')
mark4 = mlines.Line2D([], [], color='m',linestyle = 'None', marker='d', markersize=30, label='Blue stars')
mark5 = mlines.Line2D([], [], color='g',linestyle = 'None', marker='*', markersize=30, label='Blue stars')
mark6 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='p', markersize=30, label='Blue stars')
mark7 = mlines.Line2D([], [], color='m',linestyle = 'None', marker='h', markersize=30, label='Blue stars')
mark8 = mlines.Line2D([], [], color='c',linestyle = 'None', marker='X', markersize=30, label='Blue stars')


mark9 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='^', markersize=30, label='Blue stars', mec='k', mew=c, mfc='w')
mark10 = mlines.Line2D([], [], color='y',linestyle = 'None', marker='o', markersize=30, label='Blue stars',mec='y', mew=c, mfc='w')
mark11 = mlines.Line2D([], [], color='c',linestyle = 'None', marker='s', markersize=30, label='Blue stars',mec='c', mew=c, mfc='w')
mark12 = mlines.Line2D([], [], color='m',linestyle = 'None', marker='d', markersize=30, label='Blue stars',mec='m', mew=c, mfc='w')
mark13 = mlines.Line2D([], [], color='g',linestyle = 'None', marker='*', markersize=30, label='Blue stars', mec='g', mew=c, mfc='w')
mark14 = mlines.Line2D([], [], color='k',linestyle = 'None', marker='p', markersize=30, label='Blue stars', mec='k', mew=c, mfc='w')
mark15 = mlines.Line2D([], [], color='m',linestyle = 'None', marker='h', markersize=30, label='Blue stars', mec='m', mew=c, mfc='w')


aa = 6
line1 = mlines.Line2D([], [], color='k', linestyle = '--', linewidth = aa )
line2 = mlines.Line2D([], [], color='k', linestyle = '-', linewidth = aa )
line3 = mlines.Line2D([], [], color='k', linestyle = '-.', linewidth =aa )
line4 = mlines.Line2D([], [], color='k', linestyle = ':', linewidth = aa)



plt.grid()

plt.tick_params(axis='x', labelsize=30)
plt.xlabel('Epoch',fontsize=30)
plt.xticks([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0],
           ['1','','2','','3','','4','','5'])
plt.ylabel('Accuracy and Loss ', fontsize = 30)
plt.tick_params(axis='y', labelsize=30)
first_legend = plt.legend(handles = [mark1,mark2,mark3,mark4,mark5,mark6,mark7,mark8,mark9,mark10,mark11,mark12,mark13,mark14,mark15],
                           labels = ('Threshold 69','Threshold 70','Threshold 71', 'Threshold 72', 'Threshold 73', 'Threshold 74','Threshold 75','Threshold 76','Threshold 77','Threshold 78','Threshold 79','Threshold 80','Threshold 81','Threshold 82','Threshold 83'),
                           prop={'size': 30}, bbox_to_anchor=(1, 0.25), loc='lower left')
ax = plt.gca().add_artist(first_legend)
plt.legend(handles = [line1,line2,line3,line4],
           labels = ('Train loss','Train accuracy','Validation loss','Validation accuracy'),
           prop={'size': 30}, bbox_to_anchor=(1, 0.9), loc='lower left')
plt.grid()

plt.show()
#VGG = pd.read_csv('E:\새 폴더\graph\original\TL_DA_VGG19.csv')

