from convert_image_to_npy import Prediction_Contours
from image_contour import exe_Contours
from image_division import random_division
from training import CNN
from Predict import Prediction
from all_result import plot_and_chart
import os


os.environ["CUDA_VISIBLE_DEVICES"]= '2'

def execute(class_name,pred_thr,train_thr,size,train_ratio,model,epoch,show=False,image_save=True):
    '''
    for i in range(len(pred_thr)):
        x = pred_thr[i]
        Prediction_Contours(x,size,show=show,image_save=image_save)
    '''
    '''
    for i in range(len(train_thr)):
        x = train_thr[i]
        exe_Contours(x,size)
    random_division(train_ratio)
    '''
    
    
    for i in range(len(model)):
        #CNN(train_thr = train_thr, model=model[i], epoch=epoch)
        Prediction(model=model[i], class_name=class_name, save_only_csv = False, Grad_CAM = True)
        plot_and_chart(model[i])
    
        
        
pred_thr = [25,30,35,40,45]
train_thr = [68,84]
model = ['resnet-152v2']
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