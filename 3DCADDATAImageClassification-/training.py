from keras.applications.vgg19 import VGG19
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.densenet import DenseNet201
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,Dropout,Dense
from keras.callbacks import ModelCheckpoint
from keraspp import skeras
from keras import backend as K
import natsort
import matplotlib.pyplot as plt
import pandas as pd
import os

def build_finetune_model(base_model,nb_classes):
    for layer in base_model.layers[:5]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    finetune_model = Model(input=base_model.input, output=predictions)
    # finetune_model.summary()
    return finetune_model

def plot_training(history, accuracy_save_path, loss_save_path):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and Validation accuracy')
    fig1 = plt.gcf()
    fig1.savefig(accuracy_save_path)
    plt.close(fig1)
    
    plt.plot(epochs, loss, 'b')
    plt.plot(epochs, val_loss, 'r')
    plt.title('Training and validation loss')
    fig2 = plt.gcf()
    fig2.savefig(loss_save_path)
    plt.close(fig2)

def CNN(train_thr, model,epoch=5):
    # cuda.close()
    # cuda.select_device(gpu)

    image_size = 224
    class_list = ['B1','B2','B3']
    NUM_EPOCH = epoch  
    
    if model == 'vgg-19':
        base_model = VGG19(include_top = False, 
                           weights = 'imagenet', 
                           input_shape = (image_size,image_size,3))
        BATCH_SIZE = 64
    if model == 'resnet-152v2':
        base_model = ResNet152V2(include_top = False,
                                 weights = 'imagenet', 
                                 input_shape = (image_size,image_size,3))
        BATCH_SIZE = 32
    if model == 'densenet-201':
        base_model = DenseNet201(include_top = False, 
                                 weights = 'imagenet', 
                                 input_shape = (image_size,image_size,3))
        BATCH_SIZE = 16

    finetune_model = build_finetune_model(base_model, nb_classes = len(class_list))

    # thresholding 임계값 x를 기준으로 나눠진 이미지셋의 train과 test 경로 찾아서 list 형식으로 append
    imageset_path = './con_image'
    imageset_path_dir_list = []
    for i in range(len(train_thr)):
        imageset_path_dir_name = '%s%s' % ('thr_',train_thr[i]) #EX. thr_75
        imageset_path_dir_list.append(imageset_path_dir_name) #EX. ['thr_75',..., 'thr_76']
    imageset_path_dir_list = natsort.natsorted(imageset_path_dir_list, reverse = False) #EX. ['thr_75',..., 'thr_76']
    print(imageset_path_dir_name)
    print(imageset_path_dir_list)
    train_dir = []
    test_dir = []
    for i in range(len(imageset_path_dir_list)):
        target_imageset_path = '%s%s%s' % (imageset_path,'/',imageset_path_dir_list[i]) #Ex. ./con_image/thr_76
        target_imageset_dir = os.listdir(target_imageset_path) 
        target_imageset_dir = natsort.natsorted(target_imageset_dir, reverse = False) #EX. ['test','train']
        train_data_dir = '%s%s%s' % (target_imageset_path,'/',target_imageset_dir[1]) #EX. ./con_image/thr_76/train
        test_data_dir = '%s%s%s' % (target_imageset_path,'/',target_imageset_dir[0]) #EX. ./con_image/thr_76/test
        
        train_dir.append(train_data_dir) #Ex. ['./con_image/thr_76/train','./con_image/thr_80/train',...]
        test_dir.append(test_data_dir) #Ex. ['./con_image/thr_76/test','./con_image/thr_80/test',...]
    
    # 훈련 시작
    for route in range(len(train_dir)):
        
        print('apple')
        
        #train과 test 이미지셋 경로 지정
        train_path = train_dir[route]
        test_path = test_dir[route]
        
        train_dataget = ImageDataGenerator(rescale = 1./255)
        train_generator = train_dataget.flow_from_directory(train_path,
                                                            target_size = (image_size, image_size),
                                                            batch_size = BATCH_SIZE,
                                                            class_mode = 'categorical')
        test_datagen = ImageDataGenerator(rescale = 1./255)
        test_generator = test_datagen.flow_from_directory(test_path,
                                                                target_size = (image_size, image_size),
                                                                batch_size = BATCH_SIZE,
                                                                class_mode = 'categorical')
        
        train_amount = os.listdir(train_path+'/'+class_list[0]) #EX. ./con_image/thr_76/B1/~.jpg
        train_amount = len(train_amount) * len(class_list)
        test_amount = os.listdir(test_path+'/'+class_list[0]) #EX. ./con_image/thr_76/B1/~.jpg
        test_amount = len(test_amount) * len(class_list)
        
        num_train_images = train_amount
        num_test_images = test_amount
        
        finetune_model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr = 2e-5), metrics=['acc'])

        save_folder_name = '%s%s%s' % (imageset_path_dir_list[route],'_',model) #EX. thr76_vgg-19
        save_folder_path = '%s%s%s%s' % ('./',model,'/',save_folder_name) #EX. ./vgg-19/thr_76_vgg-19
        if not(os.path.isdir(os.path.join(save_folder_path))):
            os.makedirs(os.path.join(save_folder_path))
        filepath = '%s%s%s%s%s' % (save_folder_path,'/{epoch:02d}','_',imageset_path_dir_list[route],'.hdf5') #EX. ./vgg-19/thr_76_vgg-19/01_thr76.hdf5
        checkpoint = ModelCheckpoint(filepath, monitor = ['val_acc'], verbose = 1, mode = 'max')
        callbacks_list = [checkpoint]

        history = finetune_model.fit_generator(train_generator,
                                               epochs = epoch,
                                               steps_per_epoch=num_train_images // BATCH_SIZE,
                                               shuffle = True,
                                               callbacks = callbacks_list,
                                               validation_data = test_generator,
                                               nb_val_samples = num_test_images)

        skeras.save_history_history(save_folder_name, history.history, fold = save_folder_path) #EX. ./vgg-19/thr_76_vgg-19/thr76_vgg-19.npy
        
        score = finetune_model.evaluate(test_generator, verbose = 0)
        hist_df = pd.DataFrame(history.history)
        
        hist_json_file = '%s%s%s%s%s%s' % (save_folder_path,'/',imageset_path_dir_list[route],'_',model,'.json') #EX. ./vgg-10/thr_76_vgg-19/thr76_vgg-19.json
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)
            
        hist_csv_file = '%s%s%s%s%s%s' % (save_folder_path,'/',imageset_path_dir_list[route],'_',model,'.csv') #EX. ./vgg-10/thr_76_vgg-19/thr76_vgg-19.csv
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
            
        plot_save_path_1 = '%s%s%s%s%s%s%s' % (save_folder_path,'/',imageset_path_dir_list[route],'_',model,'_accuracy','.png') #EX. .../thr76_vgg-19_accuracy.png
        plot_save_path_2 = '%s%s%s%s%s%s%s' % (save_folder_path,'/',imageset_path_dir_list[route],'_',model,'_loss','.png') #EX. .../thr76_vgg-19_loss.png
        plot_training(history,plot_save_path_1,plot_save_path_2)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        