import os
import numpy as np
import natsort
import cv2

def Prediction_Contours(x,size,show=False,image_save=False):
    
    #./predictions/image 폴더 안의 폴더들에 대한 경로를 리스트 형태로 불러들임
    path = './prediction/image'
    predict_image_folder_list = os.listdir(path)
    predict_image_folder_list = natsort.natsorted(predict_image_folder_list, reverse = False) #EX. ['B1','B2','B3']
    
    X = []
    y = []
    #각 폴더 안의 이미지들의 이름을 리스트 형태로 불러들임 (.png & .jpg)
    for i in range(len(predict_image_folder_list)):
        predict_image_folder_dir = '%s%s%s' % (path,'/',predict_image_folder_list[i]) #EX. './prediction/image/B1'
        predict_image_list = os.listdir(predict_image_folder_dir)
        predict_image_list = natsort.natsorted(predict_image_list, reverse = False) #EX. ['B1~.jpg',...,'B1~.png']
        for j in range(len(predict_image_list)):
            load_image_path = '%s%s%s' % (predict_image_folder_dir,'/',predict_image_list[j]) #EX. './prediction/image/B1/B1~.jpg'
            img_con = cv2.imread(load_image_path)
            imgray = cv2.cvtColor(img_con, cv2.COLOR_BGR2GRAY)
            ret, thr = cv2.threshold(imgray, x, 255, cv2.THRESH_BINARY)  # White Bakckground, Black Object #Prediction = 100, Training = 76
            contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_con, contours, -1, (0, 255, 0), 1)
            thr_resize = cv2.resize(thr, (size,size)) #이미지를 size argument에 맞춰서 크기 조정
            
            save_folder_name = '%s%s%s%s' % ('predict_thr_',x,'_size_',size) #EX. predict_thr_76_size_224
            save_folder_path = '%s%s%s%s' % ('./prediction/thr_image/',save_folder_name,'/',predict_image_folder_list[i]) #EX. ./prediction/thr_image/predict_thr_76_size_224/B1
            if not(os.path.isdir(os.path.join(save_folder_path))):
                os.makedirs(os.path.join(save_folder_path))
            save_image_name = '%s%s%s%s%s%s' % ('thr_',x,'_size_',size,'_',predict_image_list[j]) #EX. thr_76_size_224_B1~.jpg
            thr_resize_image_name = '%s%s%s' % (save_folder_path,'/',save_image_name) 
            cv2.imwrite(thr_resize_image_name, thr_resize)
            
            thr_image = cv2.imread(thr_resize_image_name,cv2.IMREAD_COLOR)

            X.append(thr_image/255)
            y.append(i)
            
            if show == True:
                cv2.imshow('thr', thr)
                cv2.imshow('thr_resize', thr_resize)
                cv2.imshow('contour', img_con)
                cv2.waitkey(0)
                cv2.destroyAllWindows()
            
            '''
            # image_save가 True일 경우 ./prediction/thr_image/thr_76_size_224경로에 이미지 저장됨
            if image_save == True:
                save_folder_name = '%s%s%s%s' % ('predict_thr_',x,'_size_',size) #EX. predict_thr_76_size_224
                save_folder_path = '%s%s%s%s' % ('./prediction/thr_image/',save_folder_name,'/',predict_image_folder_list[i]) #EX. ./prediction/thr_image/predict_thr_76_size_224/B1
                if not(os.path.isdir(os.path.join(save_folder_path))):
                    os.makedirs(os.path.join(save_folder_path))
                save_image_name = '%s%s%s%s%s%s' % ('thr_',x,'_size_',size,'_',predict_image_list[j]) #EX. thr_76_size_224_B1~.jpg
                thr_resize_image_name = '%s%s%s' % (save_folder_path,'/',save_image_name) 
                cv2.imwrite(thr_resize_image_name, thr_resize)
            '''
        # append 함수를 이용해 만든 리스트를 np를 이용해 행렬로 변환 후 .npy파일로 저장
        X1 = np.array(X)
        y1 = np.array(y)
        npy_save_folder_path = './prediction/npy/'
        npy_save_folder_name_X = '%s%s%s%s%s%s' % (npy_save_folder_path,'thr_',x,'_size_',size,'_X.npy')
        npy_save_folder_name_y = '%s%s%s%s%s%s' % (npy_save_folder_path,'thr_',x,'_size_',size,'_y.npy')
        np.save(npy_save_folder_name_X, X1)
        np.save(npy_save_folder_name_y, y1)
        
# Prediction_Contours(100,224,show=False,image_save=True)
        
                    
            
            
    