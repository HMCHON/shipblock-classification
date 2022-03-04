
##ffd
import xml.etree.ElementTree as ET
import numpy as np
import sys
import os
from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import numpy as np 
import matplotlib.pyplot as plt

IN = 1 #IN = Image Number
OA = 400 #OB = Object Amount
"""
Load image and xml file
"""


while (IN <= OA ):
    
    strIN = str(IN)
    zfillIN = strIN.zfill(3)
    
    Image_number = '%s%s%s' % ('IMG_',zfillIN,'.JPG')
    Image_xml = '%s%s%s' % ('IMG_',zfillIN,'.xml')
    New_Image = '%s%s%s' % ('IMG_',zfillIN,'.JPG')
    New_xml = '%s%s%s' % ('IMG_',zfillIN,'.xml')
    #print(Image_number)
    
    doc = ET.parse(Image_xml) # load xml file
    root = doc.getroot()
    img = cv2.imread(Image_number) # load Image
    img_shape =  np.array((img.shape), dtype = int )
    x = img_shape[1]
    y = img_shape[0]
    
    for size in root.iter('size'):
        width = size.find('width').text
        height = size.find('height').text
        depth = size.find('depth').text
        
    size_check = np.array((width, height, depth), dtype = int )
    a = size_check[1]
    b = size_check[0]
    
    """
    make bbox matrix
    """
    
    c = 0
    channel = 0
    on = 4 # object number
    
    
    

    bbox = np.zeros((on,5), dtype = float )
    boxes = np.zeros((on,5), dtype = float )
        
    for object in root.iter('object'):
        
        for bndbox in object.iter('bndbox'):
            x1 = bndbox.find('xmin').text
            y1 = bndbox.find('ymin').text
            x2 = bndbox.find('xmax').text
            y2 = bndbox.find('ymax').text
            
            box = np.array((x1,y1,x2,y2,channel), dtype = float )
            bbox[c, ] = box
            
            bboxes = np.array((y2,x1,y1,x2,channel), dtype = float )
            boxes[c, ] = bboxes
            
            c+=1
    """   
    Fitting boundary box to reload Image coordinate
    """
    if (a == x) and (b == y):
        bboxes = boxes
        bboxes[:,0] = x - bboxes[:,0]
        bboxes[:,2] = x - bboxes[:,2]
    else:
        bboxes = bbox
    #print('Before',bboxes)
    
    """
    data augmantation
    """
    transforms = Sequence([]) #you can change this option
    # HorizontalFlip()
    # Scale(scale_x = 0.2, scale_y = 0.2)
    # Translate(translate_x = 0.2, translate_y = 0.2, diff = False)
    # Rotate(angle)
    # Shear(shear_factor=0.2)
    # Resize(inp_dim)
    # RandomHSV(hue = None, saturation = None, brightness = None )
    # Sequence(augmentation, probs = 1)
    img, bboxes = transforms(img, bboxes)
    plt.imshow(draw_rect(img, bboxes))
    #print('After\n',bboxes)
    """
    Rewrite Image with new xml
    """
    bboxes_shape = bboxes.shape
    i = 0
    j = 0
    BOX = bboxes[:,0:4]
    I = (bboxes_shape[0] - 1)

    #Rewrite boundary box data
    while( i == I ):
        for object in root.iter('object'):
            for bndbox in object.iter('bndbox'):
                new_boxes = np.array(BOX[i,:], dtype = int)
                if (sum(new_boxes) == 0):
                    pass
                else:
                    for xmin in bndbox.iter('xmin'):
                        new_xmin = new_boxes[0]
                        xmin.text = str(new_xmin)
                    for ymin in bndbox.iter('ymin'):
                        new_ymin = new_boxes[1]
                        ymin.text = str(new_ymin)   
                    for xmax in bndbox.iter('xmax'):
                        new_xmax = new_boxes[2]
                        xmax.text = str(new_xmax)
                    for ymax in bndbox.iter('ymax'):
                        new_ymax = new_boxes[3]
                        ymax.text = str(new_ymax)
                        
                i += 1
           
    #Rewrite Image name data
    for filename in root.iter('filename'):
        filename.text = New_Image
        
    #Rewrite Object_name if it's number.
    for object in root.iter('object'):        
        for name in object.iter('name'):            
            if (name.text == '1'):
                name.text = 'A'            
            if (name.text == '2'):
                name.text = 'B'            
            if (name.text == '3'):
                name.text ='C'            
            if (name.text == '4'):
                name.text = 'D'
               
    doc.write(New_xml, encoding="utf-8", xml_declaration = True )
    cv2.imwrite(New_Image,img)
    print('Complete',Image_number,'to',New_Image,'!:D')
    
    IN = IN + 1





