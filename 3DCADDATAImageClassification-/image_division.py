import os
import natsort
import random
import shutil

def random_division(train_ratio):
    folder_path = os.path.join('./con_image')
    dir_list = os.listdir(folder_path)
    dir_list = natsort.natsorted(dir_list, reverse = False) #EX. ['thr_76','thr_80','thr_83']
    
    for i in range(len(dir_list)):
        target_folder_path = '%s%s%s' % (folder_path,'/',dir_list[i]) #Ex. ./con_image/thr_76
        target_folder_dir = os.listdir(target_folder_path)
        target_folder_dir = natsort.natsorted(target_folder_dir, reverse = False) #EX. ['B1','B2','B3']
        
        for j in range(len(target_folder_dir)):
            target_model_folder_path = '%s%s%s' % (target_folder_path,'/',target_folder_dir[j]) #EX. ./con_image/thr_76/B1
            target_model_folder_image_list = os.listdir(target_model_folder_path) #Ex. ['B1_X_0_Z_0.jpg']...
            image_file_list = target_model_folder_image_list
            #image_file_list = [file for file in target_model_folder_image_list if file.endwith(".jpg")] #EX. ['HFlip0_thr_76_size_224_B1_X_0_Z_0.jpg', ... ,'rotate90...jpg']
            image_file_list = natsort.natsorted(image_file_list, reverse = False)
            
            train_image_amount = round(len(image_file_list)  * train_ratio)
            test_image_amount = len(image_file_list) - train_image_amount
            test_image_list = random.sample(image_file_list, test_image_amount) # test_image_amount 만큼 대상 폴더 안의 이미지 랜덤으로 선정
            
            # test 폴더로 (1-train_ratio) 만큼의 이미지를 옮김
            for test in range(len(test_image_list)):
                test_folder_name = '%s%s%s' % (target_folder_path,'/test/',target_folder_dir[j]) #EX. ./con_image/thr_76/test/B1
                if not(os.path.isdir(os.path.join(test_folder_name))):
                    os.makedirs(os.path.join(test_folder_name))
                src = '%s%s%s' % (target_model_folder_path,'/',test_image_list[test]) #EX. ./con_image/thr_76/B1/~.jpg
                dst = '%s%s%s' % (test_folder_name,'/',test_image_list[test]) #EX. ./con_image/thr_76/test/B1/~.jpg
                shutil.move(src,dst)
            
            # train 폴더로 (train_ratio) 만큼의 이미지를 옮김
            train_image_list = os.listdir(target_model_folder_path) #EX. ['HFlip0_thr_76_size_224_B1_X_0_Z_0.jpg', ... ,'rotate90...jpg']
            train_image_list = [file for file in train_image_list if file.endswith(".jpg")]
            train_image_list = natsort.natsorted(train_image_list, reverse = False)
            for train in range(len(train_image_list)):
                train_folder_name = '%s%s%s' % (target_folder_path,'/train/',target_folder_dir[j]) #EX. ./con_image/thr_76/B1/train/B1
                if not(os.path.isdir(os.path.join(train_folder_name))):
                    os.makedirs(os.path.join(train_folder_name))
                src = '%s%s%s' % (target_model_folder_path,'/',train_image_list[train])  #EX. ./con_image/thr_76/B1/~.jpg
                dst = '%s%s%s' % (train_folder_name,'/',train_image_list[train]) #EX. ./con_image/thr_76/train/B1/~.jpg
                shutil.move(src,dst)
            os.removedirs(target_model_folder_path) #EX. ./con_image/thr_76/B1 폴더 삭제
        
# random_division(0.8)