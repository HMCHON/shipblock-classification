from training import CNN
from predict import Prediction
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"]= '2'

def execute(class_name,pred_thr,train_thr,size,train_ratio,model,epoch,show=False,image_save=False):
    for i in range(len(model)):
        #CNN(model=model[i],epoch=epoch)
        Prediction(model=model[i], class_name=class_name)
        
pred_thr = [75,80,85,90,95,100]
train_thr = [75,77,79,81,83]
model = ['densenet-201']
class_name = ['B1','B2','B3']

execute(class_name = class_name,
        pred_thr=pred_thr,
        train_thr=train_thr,
        size=224,
        train_ratio=0.8,
        model=model,
        epoch=5,
        show=False,
        image_save=True)