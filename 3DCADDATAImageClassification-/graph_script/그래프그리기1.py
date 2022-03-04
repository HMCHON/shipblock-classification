import pandas as pd                # 데이터를 저장하고 처리하는 패키지
import matplotlib as mpl           # 그래프를 그리는 패키지
import matplotlib.pyplot as plt

VGG = pd.read_csv('C:\\Users\\lane\\Desktop\\학위논문\\graph\\original\\TL_DA_VGG19.csv')
RES = pd.read_csv('C:\\Users\lane\Desktop\학위논문\graph\original\TL_DA_Resnet-152v2.csv')
DEN = pd.read_csv('C:\\Users\lane\Desktop\학위논문\graph\original\TL_DA_DenseNet-201.csv')

plt.rcParams["font.family"] = 'batang'
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (14,4)




V_val_loss = VGG['val_loss']
V_val_acc = VGG['val_acc']
V_loss = VGG['loss']
V_acc = VGG['acc']

R_val_loss = RES['val_loss']
R_val_acc = RES['val_acc']
R_loss = RES['loss']
R_acc = RES['acc']

D_val_loss = DEN['val_loss']
D_val_acc = DEN['val_acc']
D_loss = DEN['loss']
D_acc = DEN['acc']

plt.figure(figsize=(30,20))


a = 3
b = 15
plt.plot(V_val_loss,'k-o',linewidth = a,markersize = b, label = 'VGG19 validation loss')
plt.plot(V_loss,'k--o', linewidth = a,markersize = b,label = 'VGG19 train loss')
plt.plot(V_val_acc,'k:o', linewidth = a,markersize = b,label = 'VGG19 validation accuracy')
plt.plot(V_acc, 'k-.o',linewidth = a,markersize = b,label = 'VGG19 train accuracy')

plt.plot(R_val_loss,'m-^', linewidth = a,markersize = b,label = 'Resnet-152V2 validation loss')
plt.plot(R_loss,'m--^', linewidth = a,markersize = b,label = 'Resnet-152V2  train loss')
plt.plot(R_val_acc,'m:^',linewidth = a, markersize = b,label = 'Resnet-152V2 validation loss')
plt.plot(R_acc,'m-.^', linewidth = a,markersize = b,label = 'Resnet-152V2 train accuracy')

plt.plot(D_val_loss,'y-s', linewidth = a,markersize = b,label = 'Densenet-201 validation loss')
plt.plot(D_loss,'y--s',linewidth = a,markersize = b, label = 'Densenet-201 train loss')
plt.plot(D_val_acc,'y:s',linewidth = a,markersize = b, label = 'Densenet-201 validation accuracy')
plt.plot(D_acc,'y-.s', linewidth = a,markersize = b,label = 'Densenet-201 train accuracy')

plt.tick_params(axis='x', labelsize=30)
plt.xlabel('Epoch',fontsize=30)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10],
           [1,2,3,4,5,6,7,8,9,10])
plt.tick_params(axis='y', labelsize=30)
plt.ylabel('Accuracy and loss',fontsize=30)
plt.axis([-0.3,9.3,-0.01,1.2])
plt.grid()
plt.legend(prop={'size': 30},loc='right')
plt.show()