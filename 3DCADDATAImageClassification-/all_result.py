import os
import natsort
import csv
import pandas as pd
from pandas import DataFrame

def plot_and_chart(model):
    #Parameter  설정
    td_case_name = []
    td_name = []
    td_TP = []
    td_TN = []
    td_FP = []
    td_FN = []
    td_Accuracy = []
    td_Precision = []
    td_Recall = []
    
    case_total_Accuracy = []
    case_total_F1_Score = []
    
    
    data_dir = '%s%s' % ('./',model) #EX. ./densenet-201
    data_folder_list = os.listdir(data_dir)
    data_folder_list = natsort.natsorted(data_folder_list, reverse = False) #['thr_74_densenet-201',...,'thr_83_densenet-201']
    for i in range(len(data_folder_list)):
        target_data_folder = data_folder_list[i]
        target_data_folder_dir = '%s%s%s' % (data_dir,'/',target_data_folder) #EX. ./densenet-201/thr_74_densenet-201
        # target_data_folder_dir 안의 .hdf5파일 찾기
        count_epoch_list = os.listdir(target_data_folder_dir)
        count_epoch_list = [file for file in count_epoch_list if file.endswith('.hdf5')]
        count_epoch_list = natsort.natsorted(count_epoch_list, reverse = False) #EX. ['01_thr_74.hdf5',...,'05_thr_74.hdf5']
        count_epoch = len(count_epoch_list) # 5
        for j in range(count_epoch):
            target_data_dir = '%s%s%d' % (target_data_folder_dir,'/epoch_',j+1) #EX. ./densenet-201/thr_74_densenet-201/epoch_1
            target_data_label_folder = os.listdir(target_data_dir)
            target_data_label_folder = natsort.natsorted(target_data_label_folder) #EX. ['thr_75_size_224',...,'thr_83_size_224']
            for l in range(len(target_data_label_folder)):
                target_data_path = '%s%s%s' % (target_data_dir,'/',target_data_label_folder[l]) #EX. ./densenet-201/thr_74_densenet-201/epoch_1/thr_75_size_224
                target_data = os.listdir(target_data_path)
                target_data = [file for file in target_data if file.endswith('.csv')]
                target_data = natsort.natsorted(target_data, reverse = False) #EX. ['pred_thr_75_size_224_epoch_01_thr_74.csv']
                
                '''
                CSV 예시
                ['', 'class', 'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'Accuracy', 'Avg_F1-Score']
                ['0', 'B1', '14', '54', '6', '16', '0.7', '0.4666666666666667', '0.6', '0.6092']
                ['1', 'B2', '21', '49', '11', '9', '0.65625', '0.7', '0.6', '0.6092']
                ['2', 'B3', '19', '41', '19', '11', '0.5', '0.6333333333333333', '0.6', '0.6092']
                '''
                # print(target_data[0])
                ttd ='%s%s%s' % (target_data_path,'/', target_data[0])
                target_csv = open(ttd,'r')
                rdr = csv.reader(target_csv)
                for line in rdr:
                    if not line[1] == 'class':
                        target_data_case_name = ((target_data[0])[:-4])
                        target_data_name = line[1]
                        target_data_TP = line[2]
                        target_data_TN = line[3]
                        target_data_FP = line[4]
                        target_data_FN = line[5]
                        if int(target_data_TP) == 0:
                            target_data_Accuracy == 0
                        if int(target_data_TN) == 0:
                            target_data_Accuracy == 0
                        if not int(target_data_TP) == 0:
                            if not int(target_data_TN) == 0:
                                target_data_Accuracy = int(target_data_TP)/int(target_data_TN)
                        target_data_Precision = line[6]
                        target_data_Recall = line[7]
                        case_Accuracy = line[8]
                        case_F1_Score = line[9]
                        
                        # 각 경우에서 생성된 csv의 정보를 다음과 같이 정리함.
                        td_case_name.append(target_data_case_name)
                        td_name.append(target_data_name)
                        td_TP.append(target_data_TP)
                        td_TN.append(target_data_TN)
                        td_FP.append(target_data_FP)
                        td_FN.append(target_data_FN)
                        td_Accuracy.append(target_data_Accuracy)
                        td_Precision.append(target_data_Precision)
                        td_Recall.append(target_data_Recall)
                        
                        case_total_Accuracy.append(case_Accuracy)
                        case_total_F1_Score.append(case_F1_Score)
                target_csv.close()
    
    
    data1 = {'case':td_name,
              'block_number':td_case_name,
              'True_Positive':td_TP,
              'True_Nagative':td_TN,
              'False_Positive':td_FP,
              'False_Nagative':td_FN,
              'Precision':td_Precision,
              'Recall':td_Recall,
              'Accuracy':td_Accuracy}
    frame1 = DataFrame(data1)
    frame1_name = '%s%s%s' % ('./result/',model,'_block_result.csv')
    frame1.to_csv(frame1_name)
    
    data2 = {'case':td_case_name,
             'Accuracy':case_total_Accuracy,
             'Average_F1-Score':case_total_F1_Score}
    frame2 = DataFrame(data2)
    frame2_name = '%s%s%s' % ('./result/',model,'_all_result.csv')
    frame2.to_csv(frame2_name)
    
#plot_and_chart('densenet-201')







