import os
import natsort
import numpy as np
import cv2

def Seek_maxmin(img, Contour):
    # Definite parameter
    width = img.shape[0]
    height = img.shape[1]
    Max_x = []
    Max_y = []
    Min_x = []
    Min_y = []
    s = 0
    ls = len(Contour)
    max_x, max_y, min_x, min_y = 0, 0, 1000, 1000

    # About find maximum and minimum x,y cordinate
    while s <= ls-1:
        d, f, g = 0, 0, 0
        ld = Contour[s].shape[0]
        while d <= ld-1:
            lg = Contour[s][d][0].shape[0]
            while g <= lg-1:
                if g == 0:
                    x = Contour[s][d][0][g]
                    g = g+1
                    if x <= 10 or x >= height-10:
                        pass
                    else:
                        if max_x <= x:
                            max_x = x
                        elif x <= min_x:
                            min_x = x
                        else:
                            pass
                elif g == 1:
                    y = Contour[s][d][0][g]
                    g = g+1
                    if y<= 10 or y >= width-10:
                        pass
                    else:
                        if max_y <= y:
                            max_y = y
                        elif y <= min_y:
                            min_y = y
                        else:
                            pass
            g = 0
            d = d+1
        s = s+1

    # About append arrray sentence
    Max_x.append(max_x)
    Max_y.append(max_y)
    Min_x.append(min_x)
    Min_y.append(min_y)

    # About Change append array to numpy array
    Max_array_x = np.array(Max_x)
    Max_array_y = np.array(Max_y)
    Min_array_x = np.array(Min_x)
    Min_array_y = np.array(Min_y)
    Array = np.array((Max_array_x,Max_array_y,Min_array_x,Min_array_y))

    return Array

def Contour_Extraction(Array,img):
    # Definite parameter from Contour Array
    max_x = Array[0][0]
    min_x = Array[2][0]
    max_y = Array[1][0]
    min_y = Array[3][0]

    # Calculate circle radius
    x = max_x - min_x
    y = max_y - min_y
    r = round(((x**2 + y**2)**0.5)/2)
    r = int(r)
    x = round(max_x+min_x)/2
    x = int(x)
    y = round(max_y+min_y)/2
    y = int(y)

    # Calculate Square that tangent to a circle
    X1,Y1,X2,Y2 = x-r,y-r,x+r,y+r
    X1 = int(X1) -10
    Y1 = int(Y1) -10
    X2 = int(X2) +10
    Y2 = int(Y2) +10
    rect_array = np.array((X1,Y1, X2, Y2))
    
    # Define show argument
    return rect_array

def Crop_img(img, Rect_Array, image_name):
    # Parameter definite
    w = img.shape[1]
    h = img.shape[0]


    # About min_x condition sentence
    if Rect_Array[0] <= 0:
        Rect_Array[0] = 0
    elif Rect_Array[0] >= w:
        Rect_Array[0] = w

    # About min_y condition sentence
    if Rect_Array[1] <= 0:
        Rect_Array[1] = 0
    elif Rect_Array[1] >= h:
        Rect_Array[1] = h

    # About max_x condition sentence
    if Rect_Array[2] <= 0:
        Rect_Array[2] = 0
    elif Rect_Array[2] >= w:
        Rect_Array[2] = w

    # About max_y condition sentence
    if Rect_Array[3] <= 0:
        Rect_Array[3] = 0
    elif Rect_Array [3] >= h:
        Rect_Array[3] = h

    # Image Crop and Save sentence
    
    if Rect_Array[2]-Rect_Array[0] >= Rect_Array[3]-Rect_Array[1]:
        hhh = (Rect_Array[3]-Rect_Array[1])/2
        center = (Rect_Array[2]-Rect_Array[0])/2 + Rect_Array[0]
        Rect_Array[0] = center-hhh
        Rect_Array[2] = center+hhh
       
    crop_img = img[Rect_Array[1]:Rect_Array[3],Rect_Array[0]:Rect_Array[2]]
    
    return crop_img

def Augmentation(img, con_save_path, load_img, x, size):
    
    Rotate30 = True
    Rotate60 = True
    Rotate90 = True
    HFlip0 = True
    HFlip1 = True
    
    if Rotate30 is True:
        rows, cols = img.shape[:2]
        M30 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
        img_rotate30 = cv2.warpAffine(img, M30, (rows, cols))
        rotate30_image_name ='%s%s%s%s%s%s%s' % (con_save_path,'/rotate30_thr_',x,'_size_',size,'_',load_img)
        cv2.imwrite(rotate30_image_name, img_rotate30)
            
    if Rotate60 is True:
        rows, cols = img.shape[:2]
        M60 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 60, 1)
        img_rotate60 = cv2.warpAffine(img, M60, (rows, cols))
        rotate60_image_name ='%s%s%s%s%s%s%s' % (con_save_path,'/rotate60_thr_',x,'_size_',size,'_',load_img)
        cv2.imwrite(rotate60_image_name, img_rotate60)
        
    if Rotate90 is True:
        rows, cols = img.shape[:2]
        M90 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_rotate90 = cv2.warpAffine(img, M90, (cols, rows))
        rotate90_image_name ='%s%s%s%s%s%s%s' % (con_save_path,'/rotate90_thr_',x,'_size_',size,'_',load_img)
        cv2.imwrite(rotate90_image_name, img_rotate90)
    
    if HFlip0 is True:
        img_HFlip0 = cv2.flip(img, 0)
        HFlip0_image_name ='%s%s%s%s%s%s%s' % (con_save_path,'/HFlip0_thr_',x,'_size_',size,'_',load_img)
        cv2.imwrite(HFlip0_image_name, img_HFlip0)
    
    if HFlip1 is True:
        img_HFlip1 = cv2.flip(img, 1)
        HFlip1_image_name ='%s%s%s%s%s%s%s' % (con_save_path,'/HFlip1_thr_',x,'_size_',size,'_',load_img)
        cv2.imwrite(HFlip1_image_name, img_HFlip1)

def Contours(x,size, show=False):
    # image 폴더 안의 폴더들에 대한 경로를 리스트 형태로 불러들임.
    path = './image'
    category_path_dir = os.listdir(path) #Ex.['B1','B2','B3']
    category_path_dir = natsort.natsorted(category_path_dir, reverse=False)
    
    # 폴더 경로 안의 이미지들의 이름을 리스트 형태로 불러들임. (.png & ,jpg)
    for i in range(len(category_path_dir)):
        image_path_dir = '%s%s%s'%(path,'/',category_path_dir[i]) #Ex. ./image/B1
        image_path = os.listdir(image_path_dir) #Ex. ['B1_X_0_Z_0.jpg']...
        img_file_list_jpg = [file for file in image_path if file.endswith(".jpg")]
        img_file_list_png = [file for file in image_path if file.endswith(".png")]
        img_file_list = img_file_list_jpg + img_file_list_png
        load_img = natsort.natsorted(img_file_list,reverse=False) # 이미지를 이름순으로 정렬함
        
        for j in range(len(load_img)):
            load_path = image_path_dir + '/' + load_img[j] #Ex. ./image/B1/B1_X_0_Z_0.jpg
            img_con = cv2.imread(load_path)
            imgray = cv2.cvtColor(img_con, cv2.COLOR_BGR2GRAY)
            ret, thr = cv2.threshold(imgray, x, 255, cv2.THRESH_BINARY)  # White Bakckground, Black Object #Prediction = 100, Training = 76
            contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_con, contours, -1, (0, 255, 0), 1)
            
            Array = Seek_maxmin(img_con, contours) #def seek_maxmin
            Rect_Array = Contour_Extraction(Array, img_con)
            Crop = Crop_img(thr, Rect_Array, image_path)
    
            if show ==True:
                cv2.imshow('thr', thr)
                cv2.imshow('contour', img_con)
                cv2.waitkey(0)
                cv2.destroyAllWindows()
                
            folder_name = '%s' % (category_path_dir[i]) #Ex. 'B1'
            con_save_path = '%s%s%s%s' %('./con_image/thr_',x,'/',folder_name)
            if not(os.path.isdir(os.path.join(con_save_path))):
                os.makedirs(os.path.join(con_save_path))
            
            Crop = cv2.resize(Crop, (size,size))
            
           
            contours_img_name = '%s%s%s%s%s%s%s' % (con_save_path,'/thr_',x,'_size_',size,'_',load_img[j])
            cv2.imwrite(contours_img_name, Crop)
            
            Augmentation(Crop, con_save_path, load_img[j], x, size) #Data Augmentation을 수행하는 함수

def exe_Contours(x,size):
    Contours(x, size)




















