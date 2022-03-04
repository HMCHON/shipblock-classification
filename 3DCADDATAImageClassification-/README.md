# 3DCADDATAImageClassification
[3차원 CAD 데이터의 multi-view image set 활용 image classification]
- 조선소의 건조 특성 상, CNN 모델의 학습에 사용될 학습이미지를 구하기 어려운 문제를 해결하기 위해, 3차원 CAD 모델을 촬영한 이미지를 학습데이터로 사용함.
- 검증에는 3차원 CAD 모델을 3D 프린터로 출력한 모형을 촬영한 이미지를 사용함.
- CNN 모델의 훈련 성능을 향상시키기 위해 전결합층을 커스텀함.
- CNN 모델의 학습 결과를 파악하기 위해 Grad-CAM 기법을 사용함.
 
    
     
     

[학습에 사용한 알고리즘 FLOW 그래프]
![포트폴리오 그림](https://user-images.githubusercontent.com/49745654/109630128-e5bb7080-7b87-11eb-8561-671c4eebe5d2.jpg)

-----------------------------------------------------------------------------------------------------------------

[Execute.py 에 입력하는 함수 스크립트]
- convert_image_to_npy.py : prediction 이미지를 npy파일로 변환하는 함수 스크립트
- image_contour : training 이미지를 npy파일로 변환하는 함수 스크립트 ( 3차원 CAD 모델 촬영 이미지에 대한 추가 전처리에 필요한 함수 포함)

<img width="356" alt="그림43" src="https://user-images.githubusercontent.com/49745654/109635838-2cac6480-7b8e-11eb-8443-492204866613.png">



- image_division : training 이미지를 변환한 npy 파일을 train과 validation 비율 대로 나눠주는 함수 스크립트
- training : CNN 모델을 학습시키는 함수 스크립트
- predict : 학습한 CNN 모델의 학습 성능을 검증하는 함수 스크립트
- all_result : confusion matrix를 이용하여 검증한 학습 성능을 가시화 하고, 수치로 산출하는 함수 스크립트.
<img width="400" alt="epoch01_prediction" src="https://user-images.githubusercontent.com/49745654/109636317-c411b780-7b8e-11eb-91f6-a5b49d9917b5.png">
<img width="400" alt="predict" src="https://user-images.githubusercontent.com/49745654/109636494-fcb19100-7b8e-11eb-90fe-7654ce6cfa02.png">

-----------------------------------------------------------------------------------------------------------------

[Execute.py 에 입력하는 변수]
- class_name : classification 객체의 이름 list
- pred_thr : predict 이미지 전처리 변수
- train_thr : train 이미지 전처리 변수
- size  : 이미지 크기
- train_ration : train과 validation 비율
- model : CNN 모델 선택
- epoch : 에포크
- show : 팝업창으로 결과 확인 여부 (True or False)
- image_save : 이미지 저장 여부 (True or False)

-----------------------------------------------------------------------------------------------------------------
[ confusion matrix 예시]

![pred_thr_90_size_224_epoch_01_thr_78_confusion_matrix npy](https://user-images.githubusercontent.com/49745654/109635610-eeaf4080-7b8d-11eb-9654-a104041c6c3d.png)


[ Grad_CAM 예시 ] 

<img width="343" alt="그림37" src="https://user-images.githubusercontent.com/49745654/109635540-da6b4380-7b8d-11eb-99a3-66a941d03ea5.png">

[ 이미지 전처리 알고리즘 FLOW 그래프]

<img width="374" alt="그림4" src="https://user-images.githubusercontent.com/49745654/109635765-16060d80-7b8e-11eb-8ac5-b1d7e04257e4.png">

