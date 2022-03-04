[ Faster-RCNN 모델의 학습을 위한 이미지의 Data augmentation에 사용되는 코드 입니다.]

Data augmentation 이점
- 획득할 수 있는 데이터의 양에 한계가 있는 경우 데이터 양을 증가시켜줄 수 있음
- 완벽하게 학습한 Neural Network 모델이 아니라면 augmentation 후의 데이터를 새로운 데이터로 인식하므로 효과적임
- 이미지의 회전, 크기 변경 등으로 효과적으로 데이터 증축이 가능 함


코드 설명
- 해당 코드는 이미지가 augmentation 될 때, 이미지의 라벨 정보를 가지고 있는 XML 파일의 정보고 같이 변경해줌


사용방법
1. shipblockimages 폴더에 augmentaion을 적용할 이미지와 이미지의 라벨 정보를 가지고 있는 XML파일을 할당함

2. editxml_while.py 파일에서 사용할 데이터셋에 맞는 변수를 입력해줌
    OA = 데이터셋에서 augmentation 할 이미지 갯수 입력
    
    transforms = Sequence([]) #you can change this option
    # HorizontalFlip()
    # Scale(scale_x = 0.2, scale_y = 0.2)
    # Translate(translate_x = 0.2, translate_y = 0.2, diff = False)
    # Rotate(angle)
    # Shear(shear_factor=0.2)
    # Resize(inp_dim)
    # RandomHSV(hue = None, saturation = None, brightness = None )
        = 적용할 augmentation 기능을 선택해 Sequence 안에 할당
    
3. augmentation 결과는 적용한 augmentation의 종류에 따라 다른 이름으로 부여되 shipblockimages 폴더에 저장됨


<img width="329" alt="그림58" src="https://user-images.githubusercontent.com/49745654/109637071-a55ff080-7b8f-11eb-911b-dfd102362ed4.png">


